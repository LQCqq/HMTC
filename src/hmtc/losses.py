"""Joint denoising, hierarchy, and cross-factor consistency objectives."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import Tensor

from hmtc.config import LossConfig
from hmtc.element_properties import COVALENT_RADII_ANGSTROM
from hmtc.models.geometry import pairwise_periodic_geometry
from hmtc.models.hmtf import HMTFOutput
from hmtc.routing import batch_routing_occupancy, off_diagonal_coassignment
from hmtc.types import CrystalState, StructureState


def periodic_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    delta = torch.remainder(prediction - target + 0.5, 1.0) - 0.5
    return delta.square()


def minimum_distance_penalty(
    element_logits: Tensor,
    frac_coords: Tensor,
    lattice: Tensor,
    active: Tensor,
    offset: float = 0.7,
    scaling_factor: float = 0.5,
) -> Tensor:
    """Penalize periodic atom pairs that are closer than a radius threshold.

    ``d_min(i,j) = (offset + r_i + r_j) * scaling_factor``.

    Expected covalent radii are computed from the element probabilities rather
    than hard argmax labels. Therefore this term is differentiable with respect
    to element logits, fractional coordinates, and lattice parameters.
    """

    num_elements = element_logits.shape[-1]
    if num_elements > len(COVALENT_RADII_ANGSTROM):
        raise ValueError(
            f"minimum-distance radii cover {len(COVALENT_RADII_ANGSTROM)} elements, "
            f"but logits contain {num_elements}"
        )
    radii = element_logits.new_tensor(COVALENT_RADII_ANGSTROM[:num_elements])
    element_probabilities = torch.softmax(element_logits, dim=-1)
    expected_radii = torch.einsum("bse,e->bs", element_probabilities, radii)
    minimum_distance = (
        offset + expected_radii[:, :, None] + expected_radii[:, None, :]
    ) * scaling_factor
    _, actual_distance = pairwise_periodic_geometry(frac_coords, lattice)
    violation = torch.relu(minimum_distance - actual_distance)

    num_sites = active.shape[1]
    unique_pairs = torch.ones(
        (num_sites, num_sites), device=active.device, dtype=torch.bool
    ).triu(diagonal=1)
    pair_mask = active[:, :, None] & active[:, None, :] & unique_pairs[None]
    return (violation.square() * pair_mask).sum() / pair_mask.sum().clamp_min(1)


def composition_distribution_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Compare both normalized formula fractions and total atom fraction."""

    predicted_size = prediction.sum(dim=-1, keepdim=True)
    target_size = target.sum(dim=-1, keepdim=True)
    predicted_formula = prediction / predicted_size.clamp_min(1.0e-7)
    target_formula = target / target_size.clamp_min(1.0e-7)
    formula = 0.5 * (predicted_formula - target_formula).abs().sum(dim=-1)
    size = F.smooth_l1_loss(
        predicted_size,
        target_size,
        reduction="none",
    ).squeeze(-1)
    return (formula + size).mean()


def element_count_loss(
    element_logits: Tensor,
    existence: Tensor,
    target_composition: Tensor,
    site_mask: Tensor,
    max_sites: int,
) -> Tensor:
    """Match the formula implied by site predictions to the target formula.

    Dataset compositions contain element counts divided by ``max_sites``.
    Converting both sides to normalized formula fractions avoids the very small
    gradients produced by a 100-dimensional element-wise mean, while site
    existence remains responsible for the absolute number of atoms.
    """

    element_probabilities = torch.softmax(element_logits, dim=-1)
    site_weight = existence * site_mask.to(existence.dtype)
    predicted_counts = torch.einsum(
        "bs,bse->be", site_weight, element_probabilities
    )
    predicted_composition = predicted_counts / float(max_sites)
    return composition_distribution_loss(predicted_composition, target_composition)


def element_denoising_loss(
    element_logits: Tensor,
    target_elements: Tensor,
    active: Tensor,
    element_mask: Tensor | None = None,
) -> Tensor:
    """Cross-entropy on corrupted active sites, or all active sites if omitted."""

    element_raw = F.cross_entropy(
        element_logits.transpose(1, 2), target_elements, reduction="none"
    )
    supervised = active if element_mask is None else active & element_mask.bool()
    return (element_raw * supervised).sum() / supervised.sum().clamp_min(1)


def _routing_losses(
    output: HMTFOutput,
    active: Tensor,
    redundancy_margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    zero = output.energy.new_zeros(())
    if not output.traces:
        return zero, zero, zero
    balance_terms = []
    entropy_terms = []
    redundancy_terms = []
    for trace in output.traces:
        for name, assignments in trace.assignments.items():
            masses = trace.masses[name]
            for level_index, assignment in enumerate(assignments):
                child_weight = (
                    active.to(assignment.dtype)
                    if level_index == 0
                    else masses[level_index - 1].detach()
                )
                # Pool routing mass over the whole mini-batch. Requiring every
                # individual crystal to use every group conflicts with sharp
                # routing whenever a structure has fewer children than groups.
                occupancy = batch_routing_occupancy(assignment, child_weight)
                uniform = torch.full_like(occupancy, 1.0 / occupancy.shape[-1])
                balance_terms.append((occupancy - uniform).square().mean())
                normalizer = child_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-7)
                entropy_per_child = -(
                    assignment.clamp_min(1.0e-8).log() * assignment
                ).sum(dim=-1)
                entropy_per_sample = (
                    entropy_per_child * child_weight
                ).sum(dim=1) / normalizer.squeeze(-1)
                # Minimize positive entropy for sharper per-site routing. The
                # balance loss separately prevents every site choosing one group.
                entropy_terms.append(entropy_per_sample.mean())

        tree_items = list(trace.assignments.items())
        for (left_name, left_levels), (right_name, right_levels) in combinations(
            tree_items, 2
        ):
            for level_index, (left, right) in enumerate(
                zip(left_levels, right_levels, strict=False)
            ):
                if left.shape[1] != right.shape[1]:
                    continue
                if level_index == 0:
                    child_weight = active.to(left.dtype)
                else:
                    left_mass = trace.masses[left_name][level_index - 1]
                    right_mass = trace.masses[right_name][level_index - 1]
                    child_weight = (0.5 * (left_mass + right_mass)).detach()
                left_coassignment = off_diagonal_coassignment(left, child_weight)
                right_coassignment = off_diagonal_coassignment(right, child_weight)
                if left_coassignment.shape[-1] == 0:
                    continue
                similarity = F.cosine_similarity(
                    left_coassignment, right_coassignment, dim=-1
                )
                redundancy_terms.append(
                    torch.relu(similarity - redundancy_margin).square().mean()
                )
    balance = torch.stack(balance_terms).mean() if balance_terms else zero
    entropy = torch.stack(entropy_terms).mean() if entropy_terms else zero
    redundancy = torch.stack(redundancy_terms).mean() if redundancy_terms else zero
    return balance, entropy, redundancy


def hmtf_loss(
    output: HMTFOutput,
    target: CrystalState,
    config: LossConfig,
    energy_evaluator: Callable[[StructureState], Tensor] | None = None,
    element_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    site_mask = target.resolved_site_mask()
    active = site_mask & (target.existence > 0.5)
    energy = F.smooth_l1_loss(output.energy, target.energy)
    lattice = F.smooth_l1_loss(output.lattice, target.lattice)
    distance = minimum_distance_penalty(
        output.element_logits,
        output.frac_coords,
        output.lattice,
        active,
        offset=config.min_distance_offset,
        scaling_factor=config.min_distance_scaling_factor,
    )
    composition = composition_distribution_loss(
        output.composition,
        target.composition,
    )
    element = element_denoising_loss(
        output.element_logits,
        target.elements,
        active,
        element_mask,
    )
    element_count = element_count_loss(
        output.element_logits,
        output.existence,
        target.composition,
        site_mask,
        target.max_sites,
    )
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
    composition_consistency = composition_distribution_loss(
        output.composition,
        site_composition,
    )
    if energy_evaluator is None:
        energy_consistency = output.energy.new_zeros(())
    else:
        evaluated = energy_evaluator(output.as_structure(target.site_mask))
        energy_consistency = F.smooth_l1_loss(output.energy, evaluated)
    routing_balance, routing_entropy, tree_redundancy = _routing_losses(
        output,
        active,
        config.tree_redundancy_margin,
    )
    terms = {
        "energy": energy,
        "lattice": lattice,
        "distance": distance,
        "composition": composition,
        "element": element,
        "element_count": element_count,
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
