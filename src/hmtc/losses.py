"""Joint denoising, hierarchy, and cross-factor consistency objectives."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import Tensor

from hmtc.config import LossConfig
from hmtc.models.hmtf import HMTFOutput
from hmtc.types import CrystalState, StructureState


def periodic_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    delta = torch.remainder(prediction - target + 0.5, 1.0) - 0.5
    return delta.square()


def _routing_losses(output: HMTFOutput) -> tuple[Tensor, Tensor, Tensor]:
    zero = output.energy.new_zeros(())
    if not output.traces:
        return zero, zero, zero
    balance_terms = []
    entropy_terms = []
    redundancy_terms = []
    for trace in output.traces:
        for assignments in trace.assignments.values():
            for assignment in assignments:
                occupancy = assignment.mean(dim=1)
                uniform = torch.full_like(occupancy, 1.0 / occupancy.shape[-1])
                balance_terms.append((occupancy - uniform).square().mean())
                entropy = -(assignment.clamp_min(1.0e-8).log() * assignment).sum(dim=-1).mean()
                entropy_terms.append(-entropy)
        first_level = {
            name: assignments[0] for name, assignments in trace.assignments.items() if assignments
        }
        for left, right in combinations(first_level.values(), 2):
            left_coassignment = torch.matmul(left, left.transpose(-1, -2)).flatten(1)
            right_coassignment = torch.matmul(right, right.transpose(-1, -2)).flatten(1)
            similarity = F.cosine_similarity(left_coassignment, right_coassignment, dim=-1)
            redundancy_terms.append(torch.relu(similarity - 0.85).mean())
    balance = torch.stack(balance_terms).mean() if balance_terms else zero
    entropy = torch.stack(entropy_terms).mean() if entropy_terms else zero
    redundancy = torch.stack(redundancy_terms).mean() if redundancy_terms else zero
    return balance, entropy, redundancy


def hmtf_loss(
    output: HMTFOutput,
    target: CrystalState,
    config: LossConfig,
    energy_evaluator: Callable[[StructureState], Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    site_mask = target.resolved_site_mask()
    active = site_mask & (target.existence > 0.5)
    energy = F.smooth_l1_loss(output.energy, target.energy)
    lattice = F.smooth_l1_loss(output.lattice, target.lattice)
    composition = F.smooth_l1_loss(output.composition, target.composition)
    element_raw = F.cross_entropy(
        output.element_logits.transpose(1, 2), target.elements, reduction="none"
    )
    element = (element_raw * active).sum() / active.sum().clamp_min(1)
    coordinate_raw = periodic_squared_error(output.frac_coords, target.frac_coords).sum(dim=-1)
    coordinate = (coordinate_raw * active).sum() / active.sum().clamp_min(1)
    existence_raw = F.binary_cross_entropy_with_logits(
        output.existence_logits, target.existence, reduction="none"
    )
    existence = (existence_raw * site_mask).sum() / site_mask.sum().clamp_min(1)
    site_composition = (
        torch.einsum("bk,bke->be", output.existence, torch.softmax(output.element_logits, dim=-1))
        / target.max_sites
    )
    composition_consistency = F.smooth_l1_loss(output.composition, site_composition)
    if energy_evaluator is None:
        energy_consistency = output.energy.new_zeros(())
    else:
        evaluated = energy_evaluator(output.as_structure(target.site_mask))
        energy_consistency = F.smooth_l1_loss(output.energy, evaluated)
    routing_balance, routing_entropy, tree_redundancy = _routing_losses(output)
    terms = {
        "energy": energy,
        "lattice": lattice,
        "composition": composition,
        "element": element,
        "coordinate": coordinate,
        "existence": existence,
        "energy_consistency": energy_consistency,
        "composition_consistency": composition_consistency,
        "routing_balance": routing_balance,
        "routing_entropy": routing_entropy,
        "tree_redundancy": tree_redundancy,
    }
    total = sum(getattr(config, name) * value for name, value in terms.items())
    return total, terms
