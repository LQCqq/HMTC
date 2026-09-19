"""Portable JSONL adapter for the initial training CLI."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from hmtc.data.schema import CrystalRecord, NormalizationStats


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
