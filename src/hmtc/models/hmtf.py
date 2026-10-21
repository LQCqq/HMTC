"""End-to-end initial HMTF crystal generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from hmtc.config import DiffusionConfig, HMTFConfig
from hmtc.diffusion.schedules import coordinate_sigma
from hmtc.manifolds.lorentz import expmap0, logmap0
from hmtc.models.cross_tree import CrossTreeTrace
from hmtc.models.embeddings import FactorEncoder, SinusoidalEmbedding
from hmtc.models.encoder_decoder import CrossTreeLatentCore, MultiTreeDecoder, MultiTreeEncoder
from hmtc.models.geometry import EquivariantCoordinateHead, PeriodicMessageLayer
from hmtc.types import CrystalState, StructureState

if TYPE_CHECKING:
    from hmtc.data.schema import NormalizationStats


@dataclass
class HMTFOutput:
    energy: Tensor
    lattice: Tensor
    composition: Tensor
    element_logits: Tensor
    frac_coords: Tensor
    existence_logits: Tensor
    traces: list[CrossTreeTrace]
    fixed_existence: Tensor | None = None
    coordinate_score: Tensor | None = None
    coordinate_sigma: Tensor | None = None
    noise_time: Tensor | None = None
    geometry_image_limit_hit: Tensor | None = None

    @property
    def existence(self) -> Tensor:
        return self.fixed_existence if self.fixed_existence is not None else torch.sigmoid(self.existence_logits)

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
    """Count-conditioned denoiser with scalar trees and a physical vector readout."""

    factor_names = ("energy", "lattice", "composition", "size")

    def __init__(
        self,
        config: HMTFConfig,
        diffusion_config: DiffusionConfig | None = None,
        normalization: NormalizationStats | None = None,
    ) -> None:
        super().__init__()
        from hmtc.data.schema import NormalizationStats

        self.config = config
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.normalization = normalization or NormalizationStats(0.0, 1.0)
        self.register_buffer("lattice_mean", torch.tensor(self.normalization.lattice_mean))
        self.register_buffer("lattice_std", torch.tensor(self.normalization.lattice_std))
        hidden = config.hidden_dim
        self.mask_token = config.num_elements
        self.time_embedding = SinusoidalEmbedding(hidden)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.element_embedding = nn.Embedding(config.num_elements + 1, hidden)
        self.count_embedding = nn.Embedding(config.max_sites + 1, hidden)
        # Absolute fractional coordinates are deliberately excluded. Geometry
        # enters only through periodic relative displacements and distances.
        site_input_dim = hidden + 1 + hidden
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
        self.element_condition = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.element_head = self._head(hidden * 2, config.num_elements)
        self.physical_refinement = nn.ModuleList([
            PeriodicMessageLayer(hidden, config.num_rbf, config.rbf_cutoff)
            for _ in range(config.geometry_refinement_layers)
        ])
        self.coordinate_head = EquivariantCoordinateHead(hidden * 2, config.num_rbf, config.rbf_cutoff)

    @staticmethod
    def _head(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def _encode_factors(self, state: CrystalState, time_tangent: Tensor) -> Tensor:
        size = state.num_atoms.to(state.energy.dtype)[:, None] / self.config.max_sites
        values = {
            "energy": state.energy,
            "lattice": (state.lattice - self.lattice_mean) / self.lattice_std,
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
        active = state.physical_site_mask()
        time_tangent = self.time_projection(self.time_embedding(time))
        count_tangent = self.count_embedding(state.num_atoms)
        element_features = self.element_embedding(torch.where(active, state.elements, torch.zeros_like(state.elements)))
        site_time = (time_tangent + count_tangent)[:, None, :].expand(-1, state.max_sites, -1)
        site_input = torch.cat(
            (element_features, active.to(state.energy.dtype).unsqueeze(-1), site_time),
            dim=-1,
        )
        sites = expmap0(self.site_encoder(site_input))
        factors = self._encode_factors(state, time_tangent)
        workspace_tangent = self.global_tokens[None].expand(state.batch_size, -1, -1)
        workspace_tangent = workspace_tangent + (time_tangent + count_tangent)[:, None, :]
        workspace = expmap0(workspace_tangent)
        site_mask = active
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
        for layer in self.physical_refinement:
            sites = layer(sites, state.frac_coords, state.lattice, state.existence,
                          active, encoded.graph)
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

        energy = self.energy_head(energy_context)
        lattice = self.lattice_head(lattice_context) * self.lattice_std + self.lattice_mean
        chemical_root = latent.roots.get("chemical")
        chemical_tangent = (
            global_tangent if chemical_root is None else logmap0(chemical_root)
        )
        element_condition = self.element_condition(
            torch.cat((factor_tangent[:, 2], chemical_tangent), dim=-1)
        )
        element_site_readout = torch.cat(
            (site_tangent + element_condition[:, None, :], global_for_sites), dim=-1
        )
        element_logits = self.element_head(element_site_readout)
        normalized_noise = self.coordinate_head(site_readout, encoded.graph, state.lattice)
        sigma = coordinate_sigma(time, self.diffusion_config.coordinate_sigma_min,
                                 self.diffusion_config.coordinate_sigma_max)
        score = -normalized_noise / sigma[:, None, None].clamp_min(1.0e-8)
        score = torch.where((sigma > 0)[:, None, None] & active[..., None], score,
                            torch.zeros_like(score))
        # A local Tweedie estimate is exposed for low-noise geometry penalties.
        # Sampling uses score with the matched VE predictor, not this estimate
        # as an arbitrary interpolation target.
        frac_coords = torch.remainder(state.frac_coords + sigma[:, None, None].square() * score, 1.0)
        frac_coords = torch.where(active[..., None], frac_coords, torch.zeros_like(frac_coords))
        existence_logits = torch.where(active, energy.new_tensor(20.0), energy.new_tensor(-20.0))
        predicted_size = state.num_atoms.to(energy.dtype)[:, None] / self.config.max_sites
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
            fixed_existence=active.to(energy.dtype),
            coordinate_score=score,
            coordinate_sigma=sigma,
            noise_time=time,
            geometry_image_limit_hit=encoded.graph.image_limit_hit,
        )
