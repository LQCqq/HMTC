"""Portable JSONL adapter for the initial training CLI."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from hmtc.data.schema import CrystalRecord, NormalizationStats
from hmtc.models.geometry import lattice_matrix_to_params


class JsonlCrystalDataset(Dataset[CrystalRecord]):
    """Load small/medium preprocessed datasets without database coupling.

    Each line must contain ``formation_energy_per_atom``, ``atomic_numbers``,
    ``frac_coords``, and a standardized lower-triangular ``lattice_matrix``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[CrystalRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    self.records.append(
                        CrystalRecord(
                            formation_energy_per_atom=float(raw["formation_energy_per_atom"]),
                            atomic_numbers=torch.tensor(raw["atomic_numbers"], dtype=torch.long),
                            frac_coords=torch.tensor(raw["frac_coords"], dtype=torch.float32),
                            lattice_matrix=torch.tensor(raw["lattice_matrix"], dtype=torch.float32),
                            material_id=raw.get("material_id"),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid record at {self.path}:{line_number}: {exc}") from exc
        if not self.records:
            raise ValueError(f"dataset {self.path} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CrystalRecord:
        return self.records[index]


def compute_energy_stats(dataset: Dataset[CrystalRecord]) -> NormalizationStats:
    energies = torch.tensor(
        [dataset[index].formation_energy_per_atom for index in range(len(dataset))],
        dtype=torch.float64,
    )
    return NormalizationStats(
        energy_mean=float(energies.mean()),
        energy_std=float(energies.std(unbiased=False).clamp_min(1.0e-8)),
    )


def compute_normalization_stats(dataset: Dataset[CrystalRecord]) -> NormalizationStats:
    """Fit energy and physical lattice scales using the training split only."""
    energies = compute_energy_stats(dataset)
    matrices = torch.stack([dataset[index].lattice_matrix.double() for index in range(len(dataset))])
    canonical = torch.linalg.cholesky(matrices @ matrices.transpose(-1, -2))
    lattice = lattice_matrix_to_params(canonical)
    return NormalizationStats(
        energy_mean=energies.energy_mean,
        energy_std=energies.energy_std,
        lattice_mean=tuple(lattice.mean(dim=0).tolist()),
        lattice_std=tuple(lattice.std(dim=0, unbiased=False).clamp_min(1.0e-3).tolist()),
    )


def compute_count_prior(dataset: Dataset[CrystalRecord], max_sites: int) -> Tensor:
    """Categorical probabilities for N=1..max_sites, fitted without val/test."""
    counts = torch.tensor([dataset[index].atomic_numbers.numel() for index in range(len(dataset))])
    if bool(((counts < 1) | (counts > max_sites)).any()):
        raise ValueError("training counts exceed the configured max_sites")
    histogram = torch.bincount(counts, minlength=max_sites + 1)[1:].float()
    return histogram / histogram.sum()
