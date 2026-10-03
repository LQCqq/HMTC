"""Shared routing statistics used by losses and diagnostics."""

from __future__ import annotations

import torch
from torch import Tensor


def batch_routing_occupancy(assignment: Tensor, child_weight: Tensor) -> Tensor:
    """Return parent occupancy after pooling routing mass over the whole batch."""

    if assignment.shape[:2] != child_weight.shape:
        raise ValueError("assignment children and child_weight must have matching shapes")
    weighted_assignment = assignment * child_weight.unsqueeze(-1)
    total_mass = child_weight.sum().clamp_min(1.0e-7)
    return weighted_assignment.sum(dim=(0, 1)) / total_mass


def off_diagonal_coassignment(
    assignment: Tensor,
    child_weight: Tensor,
) -> Tensor:
    """Return flattened off-diagonal weighted co-assignment relationships.

    Diagonal entries only describe each child being co-assigned with itself and
    therefore inflate cross-tree similarity without measuring shared grouping.
    """

    if assignment.shape[:2] != child_weight.shape:
        raise ValueError("assignment children and child_weight must have matching shapes")
    scale = child_weight.clamp_min(0.0).sqrt().unsqueeze(-1)
    weighted = assignment * scale
    coassignment = torch.matmul(weighted, weighted.transpose(-1, -2))
    child_count = coassignment.shape[-1]
    off_diagonal = ~torch.eye(
        child_count,
        device=coassignment.device,
        dtype=torch.bool,
    )
    return coassignment[:, off_diagonal]
