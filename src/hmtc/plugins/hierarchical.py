"""Reusable learned soft hierarchy in Lorentz space."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hmtc.config import TreeSpec
from hmtc.manifolds.lorentz import expmap0, logmap0, pairwise_lorentz_distance, tangent_mean
from hmtc.plugins.base import TreePass, TreePlugin, TreeResult


class LearnedHierarchy(TreePlugin):
    def __init__(self, spec: TreeSpec, hidden_dim: int, view_bias: float = 0.0) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.view_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.type_embedding = nn.Parameter(torch.full((hidden_dim,), view_bias))
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(groups, hidden_dim) * 0.02) for groups in spec.groups]
        )
        self.log_temperatures = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in spec.groups]
        )
        self.parent_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 3),
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in spec.groups
            ]
        )
        self.down_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 3),
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in spec.groups
            ]
        )

    def upward(self, leaves: Tensor, leaf_mask: Tensor, context: Tensor) -> TreePass:
        context_tangent = logmap0(context).mean(dim=1)
        leaf_tangent = logmap0(leaves)
        leaf_tangent = leaf_tangent + self.view_projection(leaf_tangent)
        leaf_tangent = leaf_tangent + self.context_projection(context_tangent)[:, None, :]
        leaf_tangent = leaf_tangent + self.type_embedding
        child = expmap0(leaf_tangent)
        child_mass = leaf_mask.to(leaf_tangent.dtype)
        levels: list[Tensor] = []
        children: list[Tensor] = []
        assignments: list[Tensor] = []
        masses: list[Tensor] = []

        for prototype, log_temperature, update in zip(
            self.prototypes, self.log_temperatures, self.parent_updates, strict=True
        ):
            children.append(child)
            keys = expmap0(prototype)[None].expand(child.shape[0], -1, -1)
            logits = -pairwise_lorentz_distance(child, keys).square()
            logits = logits / log_temperature.exp().clamp_min(0.05)
            assignment = torch.softmax(logits, dim=-1)
            weighted_assignment = assignment * child_mass.unsqueeze(-1)
            parent_mass = weighted_assignment.sum(dim=1)
            aggregate = torch.einsum("bcp,bcd->bpd", weighted_assignment, logmap0(child))
            aggregate = aggregate / parent_mass.unsqueeze(-1).clamp_min(1.0e-7)
            prototype_tangent = prototype[None].expand(child.shape[0], -1, -1)
            context_expanded = context_tangent[:, None, :].expand_as(aggregate)
            parent_delta = update(
                torch.cat((aggregate, prototype_tangent, context_expanded), dim=-1)
            )
            child = expmap0(aggregate + prototype_tangent + parent_delta)
            child_mass = parent_mass
            levels.append(child)
            assignments.append(assignment)
            masses.append(parent_mass)

        root = tangent_mean(child, weights=child_mass, dim=1)
        return TreePass(
            name=self.spec.name,
            root=root,
            levels=levels,
            children=children,
            assignments=assignments,
            masses=masses,
            leaf_mask=leaf_mask,
        )

    def downward(self, state: TreePass, bridged_root: Tensor, context: Tensor) -> TreeResult:
        parent = state.levels[-1]
        root_message = logmap0(bridged_root)[:, None, :]
        parent = expmap0(logmap0(parent) + root_message)
        context_tangent = logmap0(context).mean(dim=1)

        for level_index in reversed(range(len(state.assignments))):
            assignment = state.assignments[level_index]
            original_child = state.children[level_index]
            parent_message = torch.einsum("bcp,bpd->bcd", assignment, logmap0(parent))
            original_tangent = logmap0(original_child)
            context_expanded = context_tangent[:, None, :].expand_as(original_tangent)
            delta = self.down_updates[level_index](
                torch.cat((original_tangent, parent_message, context_expanded), dim=-1)
            )
            parent = expmap0(original_tangent + parent_message + delta)

        return TreeResult(
            name=state.name,
            root=bridged_root,
            leaves=parent,
            assignments=state.assignments,
            masses=state.masses,
        )
