"""Minimal data contract independent of any one crystal database."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hmtc.models.geometry import lattice_matrix_to_params
from hmtc.types import CrystalState


@dataclass(frozen=True)
class NormalizationStats:
    energy_mean: float
    energy_std: float

    def normalize_energy(self, energy: Tensor) -> Tensor:
        return (energy - self.energy_mean) / max(self.energy_std, 1.0e-8)

    def denormalize_energy(self, normalized: Tensor) -> Tensor:
        return normalized * self.energy_std + self.energy_mean


@dataclass
class CrystalRecord:
    """One standardized primitive/conventional cell.

    Lattices must already follow a single lower-triangular cell convention.
    Atomic numbers are one-based in records and converted to zero-based model
    indices during collation.
    """

    formation_energy_per_atom: float
    atomic_numbers: Tensor
    frac_coords: Tensor
    lattice_matrix: Tensor
    material_id: str | None = None

    def validate(self, num_elements: int, max_sites: int) -> None:
        count = int(self.atomic_numbers.numel())
        if not 1 <= count <= max_sites:
            raise ValueError(f"record has {count} atoms; supported range is [1, {max_sites}]")
        if self.frac_coords.shape != (count, 3):
            raise ValueError("frac_coords must have shape [num_atoms, 3]")
        if self.lattice_matrix.shape != (3, 3):
            raise ValueError("lattice_matrix must have shape [3, 3]")
        if bool((self.atomic_numbers < 1).any()) or bool(
            (self.atomic_numbers > num_elements).any()
        ):
            raise ValueError("atomic number lies outside configured element range")


def collate_records(
    records: list[CrystalRecord],
    num_elements: int,
    max_sites: int,
    stats: NormalizationStats,
) -> CrystalState:
    if not records:
        raise ValueError("cannot collate an empty record list")
    for record in records:
        record.validate(num_elements, max_sites)
    device = records[0].frac_coords.device
    dtype = records[0].frac_coords.dtype
    batch = len(records)
    energy = torch.tensor(
        [[record.formation_energy_per_atom] for record in records], device=device, dtype=dtype
    )
    energy = stats.normalize_energy(energy)
    # Replace each arbitrary row-vector cell by the unique lower-triangular
    # cell with the same Gram matrix. Fractional coordinates remain valid and
    # every periodic Cartesian distance is preserved.
    canonical_lattices = [
        torch.linalg.cholesky(record.lattice_matrix @ record.lattice_matrix.transpose(-1, -2))
        for record in records
    ]
    lattice = torch.stack(
        [lattice_matrix_to_params(matrix) for matrix in canonical_lattices], dim=0
    )
    elements = torch.zeros(batch, max_sites, device=device, dtype=torch.long)
    frac_coords = torch.zeros(batch, max_sites, 3, device=device, dtype=dtype)
    existence = torch.zeros(batch, max_sites, device=device, dtype=dtype)
    site_mask = torch.ones(batch, max_sites, device=device, dtype=torch.bool)
    composition = torch.zeros(batch, num_elements, device=device, dtype=dtype)
    for batch_index, record in enumerate(records):
        count = int(record.atomic_numbers.numel())
        indices = record.atomic_numbers.long() - 1
        elements[batch_index, :count] = indices
        frac_coords[batch_index, :count] = torch.remainder(record.frac_coords, 1.0)
        existence[batch_index, :count] = 1.0
        composition[batch_index].scatter_add_(0, indices, torch.ones_like(indices, dtype=dtype))
    composition = composition / max_sites
    return CrystalState(
        energy=energy,
        lattice=lattice,
        composition=composition,
        elements=elements,
        frac_coords=frac_coords,
        existence=existence,
        site_mask=site_mask,
    )


def synthetic_batch(
    batch_size: int,
    max_sites: int,
    num_elements: int,
    device: torch.device | str = "cpu",
) -> CrystalState:
    """Generate shape-valid fake data for smoke tests only."""

    counts = torch.randint(2, max_sites + 1, (batch_size,), device=device)
    positions = torch.arange(max_sites, device=device)[None]
    existence = (positions < counts[:, None]).float()
    elements = torch.randint(0, num_elements, (batch_size, max_sites), device=device)
    elements = torch.where(existence.bool(), elements, torch.zeros_like(elements))
    composition = torch.zeros(batch_size, num_elements, device=device)
    composition.scatter_add_(1, elements, existence)
    composition = composition / max_sites
    lattice = torch.randn(batch_size, 6, device=device) * 0.1
    lattice[:, (0, 2, 5)] = torch.rand(batch_size, 3, device=device) + 1.0
    return CrystalState(
        energy=torch.randn(batch_size, 1, device=device),
        lattice=lattice,
        composition=composition,
        elements=elements,
        frac_coords=torch.rand(batch_size, max_sites, 3, device=device),
        existence=existence,
        site_mask=torch.ones(batch_size, max_sites, device=device, dtype=torch.bool),
    )
