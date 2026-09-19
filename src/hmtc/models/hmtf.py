"""End-to-end initial HMTF crystal generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hmtc.config import HMTFConfig
from hmtc.manifolds.lorentz import expmap0, logmap0
from hmtc.models.cross_tree import CrossTreeTrace
from hmtc.models.embeddings import FactorEncoder, FourierCoordinateEmbedding, SinusoidalEmbedding
from hmtc.models.encoder_decoder import CrossTreeLatentCore, MultiTreeDecoder, MultiTreeEncoder
from hmtc.types import CrystalState, StructureState


@dataclass
class HMTFOutput:
    energy: Tensor
    lattice: Tensor
    composition: Tensor
    element_logits: Tensor
    frac_coords: Tensor
    existence_logits: Tensor
    traces: list[CrossTreeTrace]

    @property
    def existence(self) -> Tensor:
        return torch.sigmoid(self.existence_logits)

    def as_state(self, site_mask: Tensor | None = None) -> CrystalState:
        return CrystalState(
            energy=self.energy,
            lattice=self.lattice,
            composition=self.composition,
            elements=self.element_logits.argmax(dim=-1),
            frac_coords=self.frac_coords,
            existence=self.existence,
            site_mask=site_mask,
        )

    def as_structure(self, site_mask: Tensor | None = None) -> StructureState:
        return StructureState(
            lattice=self.lattice,
            elements=self.element_logits.argmax(dim=-1),
            frac_coords=self.frac_coords,
            existence=self.existence,
            site_mask=site_mask,
        )


class HMTFModel(nn.Module):
    """Jointly denoise energy, lattice, composition, sites, elements, and coordinates."""

    factor_names = ("energy", "lattice", "composition", "size")

    def __init__(self, config: HMTFConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.mask_token = config.num_elements
        self.time_embedding = SinusoidalEmbedding(hidden)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.element_embedding = nn.Embedding(config.num_elements + 1, hidden)
        self.coordinate_embedding = FourierCoordinateEmbedding(num_bands=4)
        site_input_dim = hidden + self.coordinate_embedding.output_dim + 1 + hidden
        self.site_encoder = nn.Sequential(
            nn.Linear(site_input_dim, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.factor_encoders = nn.ModuleDict(
            {
                "energy": FactorEncoder(1, hidden),
                "lattice": FactorEncoder(6, hidden),
                "composition": FactorEncoder(config.num_elements, hidden),
                "size": FactorEncoder(1, hidden),
            }
        )
        self.factor_type = nn.Parameter(torch.randn(len(self.factor_names), hidden) * 0.02)
        self.global_tokens = nn.Parameter(torch.randn(config.num_global_tokens, hidden) * 0.02)
        self.encoder = MultiTreeEncoder(config)
        self.latent_core = CrossTreeLatentCore(config)
        self.decoder = MultiTreeDecoder(config)
        self.energy_head = self._head(hidden * 2, 1)
        self.lattice_head = self._head(hidden * 2, 6)
        self.composition_head = self._head(hidden * 2, config.num_elements)
        self.element_head = self._head(hidden * 2, config.num_elements)
        self.coordinate_head = self._head(hidden * 2, 3)
        self.existence_head = self._head(hidden * 2, 1)

    @staticmethod
    def _head(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def _encode_factors(self, state: CrystalState, time_tangent: Tensor) -> Tensor:
        site_mask = state.resolved_site_mask().to(state.existence.dtype)
        size = (state.existence * site_mask).sum(dim=1, keepdim=True) / site_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        values = {
            "energy": state.energy,
            "lattice": state.lattice,
            "composition": state.composition,
            "size": size,
        }
        tokens = []
        for index, name in enumerate(self.factor_names):
            tangent = self.factor_encoders[name](values[name]) + time_tangent
            tangent = tangent + self.factor_type[index]
            tokens.append(expmap0(tangent))
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        state: CrystalState,
        time: Tensor,
        active_trees: set[str] | None = None,
    ) -> HMTFOutput:
        state.validate(self.config.num_elements, allow_mask_token=True)
        if time.ndim == 2 and time.shape[-1] == 1:
            time = time.squeeze(-1)
        if time.shape != (state.batch_size,):
            raise ValueError("time must have shape [batch] or [batch, 1]")
        time_tangent = self.time_projection(self.time_embedding(time))
        coordinate_features = self.coordinate_embedding(state.frac_coords)
        element_features = self.element_embedding(state.elements)
        site_time = time_tangent[:, None, :].expand(-1, state.max_sites, -1)
        site_input = torch.cat(
            (element_features, coordinate_features, state.existence.unsqueeze(-1), site_time),
            dim=-1,
        )
        sites = expmap0(self.site_encoder(site_input))
        factors = self._encode_factors(state, time_tangent)
        workspace_tangent = self.global_tokens[None].expand(state.batch_size, -1, -1)
        workspace_tangent = workspace_tangent + time_tangent[:, None, :]
        workspace = expmap0(workspace_tangent)
        site_mask = state.resolved_site_mask()
        encoded = self.encoder(
            sites=sites,
            factors=factors,
            workspace=workspace,
            frac_coords=state.frac_coords,
            lattice_params=state.lattice,
            existence=state.existence,
            site_mask=site_mask,
            active_trees=active_trees,
        )
        latent = self.latent_core(encoded, workspace, factors)
        sites, decoder_trace = self.decoder(encoded, latent, self.encoder.trees)
        workspace, factors = latent.workspace, latent.factors
        traces = [CrossTreeTrace(**decoder_trace.__dict__)]

        global_tangent = logmap0(workspace).mean(dim=1)
        factor_tangent = logmap0(factors)
        site_tangent = logmap0(sites)
        global_for_sites = global_tangent[:, None, :].expand_as(site_tangent)
        site_readout = torch.cat((site_tangent, global_for_sites), dim=-1)
        energy_context = torch.cat((factor_tangent[:, 0], global_tangent), dim=-1)
        lattice_context = torch.cat((factor_tangent[:, 1], global_tangent), dim=-1)
        composition_context = torch.cat((factor_tangent[:, 2], global_tangent), dim=-1)

        energy = state.energy + self.energy_head(energy_context)
        lattice = state.lattice + self.lattice_head(lattice_context)
        element_logits = self.element_head(site_readout)
        coordinate_delta = 0.5 * torch.tanh(self.coordinate_head(site_readout))
        frac_coords = torch.remainder(state.frac_coords + coordinate_delta, 1.0)
        existence_logits = self.existence_head(site_readout).squeeze(-1)
        existence_logits = existence_logits.masked_fill(~site_mask, -20.0)
        predicted_size = (torch.sigmoid(existence_logits) * site_mask).sum(dim=1, keepdim=True)
        predicted_size = predicted_size / site_mask.sum(dim=1, keepdim=True).clamp_min(1)
        composition = torch.softmax(self.composition_head(composition_context), dim=-1)
        composition = composition * predicted_size
        return HMTFOutput(
            energy=energy,
            lattice=lattice,
            composition=composition,
            element_logits=element_logits,
            frac_coords=frac_coords,
            existence_logits=existence_logits,
            traces=traces,
        )
