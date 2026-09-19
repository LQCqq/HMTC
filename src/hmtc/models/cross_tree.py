"""The extensible cross-tree block at the center of HMTF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hmtc.config import HMTFConfig
from hmtc.manifolds.lorentz import expmap0, logmap0
from hmtc.models.attention import HyperbolicAttention
from hmtc.models.geometry import PeriodicMessageLayer
from hmtc.plugins import build_tree_plugin
from hmtc.plugins.base import TreeResult


@dataclass
class CrossTreeTrace:
    roots: dict[str, Tensor]
    assignments: dict[str, list[Tensor]]
    masses: dict[str, list[Tensor]]
    fusion_weights: Tensor


class CrossTreeBlock(nn.Module):
    """Upward trees -> typed root exchange -> workspace -> downward trees.

    Tree count is determined entirely by ``config.tree_specs``.  No operation in
    this block assumes exactly two trees.
    """

    def __init__(self, config: HMTFConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.trees = nn.ModuleDict(
            {
                spec.name: build_tree_plugin(spec, hidden)
                for spec in config.tree_specs
                if spec.leaf_entity == "site"
            }
        )
        if not self.trees:
            raise ValueError("the initial HMTF model requires at least one site-backed tree")
        self.physical = PeriodicMessageLayer(hidden, config.num_rbf, config.rbf_cutoff)
        self.root_bridge = HyperbolicAttention(hidden)
        self.workspace_attention = HyperbolicAttention(hidden)
        self.factor_attention = HyperbolicAttention(hidden)
        self.tree_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.site_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(
        self,
        sites: Tensor,
        workspace: Tensor,
        factors: Tensor,
        frac_coords: Tensor,
        lattice_params: Tensor,
        existence: Tensor,
        site_mask: Tensor,
        active_trees: set[str] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, CrossTreeTrace]:
        sites = self.physical(sites, frac_coords, lattice_params, existence, site_mask)
        context = torch.cat((workspace, factors), dim=1)
        selected = [
            (name, tree)
            for name, tree in self.trees.items()
            if active_trees is None or name in active_trees
        ]
        if not selected:
            raise ValueError("active_trees disabled every registered site tree")

        upward = {name: tree.upward(sites, site_mask, context) for name, tree in selected}
        root_stack = torch.stack([upward[name].root for name, _ in selected], dim=1)
        bridge_context = torch.cat((root_stack, workspace, factors), dim=1)
        bridged_roots = self.root_bridge(root_stack, bridge_context)
        workspace = self.workspace_attention(
            workspace, torch.cat((bridge_context, bridged_roots), dim=1)
        )
        factors = self.factor_attention(
            factors, torch.cat((workspace, bridged_roots, factors), dim=1)
        )
        downward_context = torch.cat((workspace, factors), dim=1)

        results: list[TreeResult] = []
        for index, (name, tree) in enumerate(selected):
            results.append(tree.downward(upward[name], bridged_roots[:, index], downward_context))

        site_tangent = logmap0(sites)
        leaf_tangents = torch.stack([logmap0(result.leaves) for result in results], dim=2)
        repeated_sites = site_tangent.unsqueeze(2).expand_as(leaf_tangents)
        gate_logits = self.tree_gate(torch.cat((repeated_sites, leaf_tangents), dim=-1)).squeeze(-1)
        fusion_weights = torch.softmax(gate_logits, dim=-1)
        fused = (leaf_tangents * fusion_weights.unsqueeze(-1)).sum(dim=2)
        updated_sites = expmap0(
            site_tangent + self.site_fusion(torch.cat((site_tangent, fused), dim=-1))
        )

        trace = CrossTreeTrace(
            roots={result.name: result.root for result in results},
            assignments={result.name: result.assignments for result in results},
            masses={result.name: result.masses for result in results},
            fusion_weights=fusion_weights,
        )
        return updated_sites, workspace, factors, trace
