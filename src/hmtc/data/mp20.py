"""MP-20/CDVAE CSV ingestion and conversion to the portable HMTF JSONL format."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hmtc.data.schema import CrystalRecord

ENERGY_COLUMNS = (
    "formation_energy_per_atom",
    "formation_energy_per_atom_relaxed",
    "formation_energy",
)
ID_COLUMNS = ("material_id", "mp_id", "id")
CIF_COLUMNS = ("cif", "structure", "cif_string")


@dataclass(frozen=True)
class ConversionReport:
    split: str
    input_rows: int
    written_rows: int
    skipped_too_large: int
    skipped_invalid: int


def _first(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    raise KeyError(f"none of the required columns are present: {tuple(names)}")


def _record_from_cif_row(row: dict[str, Any]) -> CrystalRecord:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "MP-20 CIF conversion requires pymatgen; install with `pip install -e '.[mp20]'`"
        ) from exc
    structure = Structure.from_str(str(_first(row, CIF_COLUMNS)), fmt="cif")
    atomic_numbers: list[int] = []
    frac_coords: list[list[float]] = []
    for site in structure:
        if not site.is_ordered:
            raise ValueError("disordered/partial-occupancy sites are not supported")
        atomic_numbers.append(int(site.specie.Z))
        frac_coords.append([float(value) for value in site.frac_coords])
    material_id = next((str(row[name]) for name in ID_COLUMNS if row.get(name)), None)
    return CrystalRecord(
        formation_energy_per_atom=float(_first(row, ENERGY_COLUMNS)),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        frac_coords=torch.tensor(frac_coords, dtype=torch.float32),
        lattice_matrix=torch.tensor(structure.lattice.matrix, dtype=torch.float32),
        material_id=material_id,
    )


def _record_as_json(record: CrystalRecord) -> dict[str, Any]:
    return {
        "material_id": record.material_id,
        "formation_energy_per_atom": record.formation_energy_per_atom,
        "atomic_numbers": record.atomic_numbers.tolist(),
        "frac_coords": record.frac_coords.tolist(),
        "lattice_matrix": record.lattice_matrix.tolist(),
    }


def convert_mp20_csv(
    source: str | Path,
    destination: str | Path,
    *,
    max_sites: int = 20,
    num_elements: int = 100,
) -> ConversionReport:
    """Convert one standard CDVAE MP-20 split without changing its split membership."""

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = written = too_large = invalid = 0
    with source.open("r", encoding="utf-8", newline="") as input_handle, destination.open(
        "w", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        for row in reader:
            total += 1
            try:
                record = _record_from_cif_row(row)
                if record.atomic_numbers.numel() > max_sites:
                    too_large += 1
                    continue
                record.validate(num_elements, max_sites)
                output_handle.write(json.dumps(_record_as_json(record), separators=(",", ":")))
                output_handle.write("\n")
                written += 1
            except (KeyError, TypeError, ValueError):
                invalid += 1
    return ConversionReport(source.stem, total, written, too_large, invalid)


def discover_mp20_splits(data_dir: str | Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    splits = {}
    for split in ("train", "val", "test"):
        candidates = (data_dir / f"{split}.csv", data_dir / f"{split}.jsonl")
        found = next((path for path in candidates if path.is_file()), None)
        if found is not None:
            splits[split] = found
    if "train" not in splits:
        raise FileNotFoundError(f"could not find train.csv or train.jsonl under {data_dir}")
    return splits
