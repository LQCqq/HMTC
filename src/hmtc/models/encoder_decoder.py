"""Explicit encoder, latent cross-tree core, and decoder for HMTF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hmtc.config import HMTFConfig
from hmtc.manifolds.lorentz import expmap0, logmap0
from hmtc.models.attention import HyperbolicAttention
from hmtc.models.geometry import PeriodicGraph, PeriodicMessageLayer, periodic_radius_graph
from hmtc.plugins import build_tree_plugin
from hmtc.plugins.base import TreePass


@dataclass
class EncodedTrees:
    sites: Tensor
    passes: dict[str, TreePass]
    roots: dict[str, Tensor]
    graph: PeriodicGraph


@dataclass
class LatentState:
    roots: dict[str, Tensor]
    workspace: Tensor
    factors: Tensor


@dataclass
class DecoderTrace:
    roots: dict[str, Tensor]
    assignments: dict[str, list[Tensor]]
    masses: dict[str, list[Tensor]]
    fusion_weights: Tensor


class MultiTreeEncoder(nn.Module):
    """Aggregate noisy shared sites upward into one sample-specific root per tree."""

    def __init__(self, config: HMTFConfig) -> None:
        super().__init__()
        self.physical = nn.ModuleList([
            PeriodicMessageLayer(config.hidden_dim, config.num_rbf, config.rbf_cutoff)
            for _ in range(config.geometry_layers)
        ])
        self.cutoff = config.rbf_cutoff
        self.image_limit = config.periodic_image_limit
        self.trees = nn.ModuleDict(
            {
                spec.name: build_tree_plugin(spec, config.hidden_dim)
                for spec in config.tree_specs
                if spec.leaf_entity == "site"
            }
        )
        if not self.trees:
            raise ValueError("MultiTreeEncoder requires at least one site-backed tree")
        self.tree_kinds = {
            spec.name: spec.kind for spec in config.tree_specs if spec.leaf_entity == "site"
        }
        # Factor order is fixed by HMTFModel.factor_names:
        # energy=0, lattice=1, composition=2, size=3.  Both trees retain the
        # shared workspace and energy/size information, while their private
        # context exposes the factor most relevant to that branch.
        self.factor_indices = {
            "chemical": (0, 2, 3),
            "geometry": (0, 1, 3),
        }

    def _tree_context(self, name: str, workspace: Tensor, factors: Tensor) -> Tensor:
        indices = self.factor_indices.get(self.tree_kinds[name])
        selected_factors = factors if indices is None else factors[:, indices]
        return torch.cat((workspace, selected_factors), dim=1)

    def forward(
        self,
        sites: Tensor,
        factors: Tensor,
        workspace: Tensor,
        frac_coords: Tensor,
        lattice_params: Tensor,
        existence: Tensor,
        site_mask: Tensor,
        active_trees: set[str] | None = None,
    ) -> EncodedTrees:
        active = site_mask & existence.bool()
        graph = periodic_radius_graph(frac_coords, lattice_params, active,
                                      self.cutoff, self.image_limit)
        for layer in self.physical:
            sites = layer(sites, frac_coords, lattice_params, existence, active, graph)
        selected = {
            name: tree
            for name, tree in self.trees.items()
            if active_trees is None or name in active_trees
        }
        if not selected:
            raise ValueError("active_trees disabled every registered site tree")
        # Count conditioning fixes the physical leaf mass throughout diffusion.
        leaf_mass = active.to(existence.dtype)
        passes = {
            name: tree.upward(
                sites,
                leaf_mass,
                self._tree_context(name, workspace, factors),
            )
            for name, tree in selected.items()
        }
        return EncodedTrees(
            sites=sites,
            passes=passes,
            roots={name: tree_pass.root for name, tree_pass in passes.items()},
            graph=graph,
        )


class CrossTreeLatentLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.root_attention = HyperbolicAttention(hidden_dim)
        self.workspace_attention = HyperbolicAttention(hidden_dim)
        self.factor_attention = HyperbolicAttention(hidden_dim)

    def forward(
        self, roots: Tensor, workspace: Tensor, factors: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        context = torch.cat((roots, workspace, factors), dim=1)
        roots = self.root_attention(roots, context)
        workspace = self.workspace_attention(
            workspace, torch.cat((roots, workspace, factors), dim=1)
        )
        factors = self.factor_attention(
            factors, torch.cat((roots, workspace, factors), dim=1)
        )
        return roots, workspace, factors


class CrossTreeLatentCore(nn.Module):
    """Exchange information among sample roots, workspace, and factor tokens."""

    def __init__(self, config: HMTFConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossTreeLatentLayer(config.hidden_dim) for _ in range(config.num_blocks)]
        )

    def forward(
        self, encoded: EncodedTrees, workspace: Tensor, factors: Tensor
    ) -> LatentState:
        names = list(encoded.roots)
        roots = torch.stack([encoded.roots[name] for name in names], dim=1)
        for layer in self.layers:
            roots, workspace, factors = layer(roots, workspace, factors)
        return LatentState(
            roots={name: roots[:, index] for index, name in enumerate(names)},
            workspace=workspace,
            factors=factors,
        )


class MultiTreeDecoder(nn.Module):
    """Broadcast updated roots down each hierarchy and fuse shared site views."""

    def __init__(self, config: HMTFConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.tree_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.site_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(
        self, encoded: EncodedTrees, latent: LatentState, trees: nn.ModuleDict
    ) -> tuple[Tensor, DecoderTrace]:
        context = torch.cat((latent.workspace, latent.factors), dim=1)
        results = [
            trees[name].downward(encoded.passes[name], latent.roots[name], context)
            for name in encoded.passes
        ]
        site_tangent = logmap0(encoded.sites)
        leaf_tangents = torch.stack([logmap0(result.leaves) for result in results], dim=2)
        repeated_sites = site_tangent.unsqueeze(2).expand_as(leaf_tangents)
        gate_logits = self.tree_gate(
            torch.cat((repeated_sites, leaf_tangents), dim=-1)
        ).squeeze(-1)
        fusion_weights = torch.softmax(gate_logits, dim=-1)
        fused = (leaf_tangents * fusion_weights.unsqueeze(-1)).sum(dim=2)
        sites = expmap0(
            site_tangent + self.site_fusion(torch.cat((site_tangent, fused), dim=-1))
        )
        return sites, DecoderTrace(
            roots={result.name: result.root for result in results},
            assignments={result.name: result.assignments for result in results},
            masses={result.name: result.masses for result in results},
            fusion_weights=fusion_weights,
        )
