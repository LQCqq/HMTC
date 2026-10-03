"""Diagnostics for learned routing hierarchies and cross-tree fusion."""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F
from torch import Tensor

from hmtc.models.hmtf import HMTFOutput
from hmtc.routing import batch_routing_occupancy, off_diagonal_coassignment


@torch.no_grad()
def routing_diagnostics(output: HMTFOutput, active: Tensor) -> dict[str, Tensor]:
    """Return per-tree routing and fusion diagnostics for one batch.

    Occupancy and entropy use the same active-site/mass weighting as the
    routing losses. Similarity compares permutation-invariant off-diagonal
    co-assignment relationships, so self-pairs cannot inflate similarity and
    renaming groups does not change the reported value.
    """

    diagnostics: dict[str, Tensor] = {}
    active = active.to(dtype=output.energy.dtype)

    for trace in output.traces:
        for name, assignments in trace.assignments.items():
            masses = trace.masses[name]
            for level_index, assignment in enumerate(assignments, start=1):
                child_weight = (
                    active if level_index == 1 else masses[level_index - 2]
                )
                occupancy = batch_routing_occupancy(assignment, child_weight)
                normalizer = child_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-7)
                entropy_per_child = -(
                    assignment.clamp_min(1.0e-8).log() * assignment
                ).sum(dim=-1)
                entropy = (
                    entropy_per_child * child_weight
                ).sum(dim=1) / normalizer.squeeze(-1)
                prefix = f"routing.{name}.level{level_index}"
                diagnostics[f"{prefix}.occupancy"] = occupancy
                diagnostics[f"{prefix}.entropy"] = entropy.mean()

        tree_items = list(trace.assignments.items())
        for (left_name, left_levels), (right_name, right_levels) in combinations(
            tree_items, 2
        ):
            for level_index, (left, right) in enumerate(
                zip(left_levels, right_levels, strict=False), start=1
            ):
                if left.shape[1] != right.shape[1]:
                    continue
                if level_index == 1:
                    child_weight = active
                else:
                    left_mass = trace.masses[left_name][level_index - 2]
                    right_mass = trace.masses[right_name][level_index - 2]
                    child_weight = 0.5 * (left_mass + right_mass)
                left_coassignment = off_diagonal_coassignment(left, child_weight)
                right_coassignment = off_diagonal_coassignment(right, child_weight)
                if left_coassignment.shape[-1] == 0:
                    continue
                similarity = F.cosine_similarity(
                    left_coassignment, right_coassignment, dim=-1
                ).mean()
                diagnostics[
                    f"routing_similarity.{left_name}__{right_name}.level{level_index}"
                ] = similarity

        tree_names = list(trace.roots)
        if trace.fusion_weights.shape[-1] != len(tree_names):
            raise ValueError("fusion weight count does not match the decoded trees")
        active_count = active.sum().clamp_min(1.0)
        for tree_index, name in enumerate(tree_names):
            fusion = trace.fusion_weights[..., tree_index]
            diagnostics[f"fusion.{name}"] = (fusion * active).sum() / active_count

    return diagnostics
