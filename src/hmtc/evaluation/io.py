"""Convert HMTC generated JSONL records to benchmark-ready CIF files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def _finite_numbers(values: Iterable[Any], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def validate_generated_record(
    record: dict[str, Any],
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    """Validate and normalize the structural fields of one generated record."""

    atomic_numbers = [int(value) for value in record["atomic_numbers"]]
    if not atomic_numbers:
        raise ValueError("atomic_numbers is empty")
    if any(number < 1 or number > 118 for number in atomic_numbers):
        raise ValueError("atomic number lies outside [1, 118]")

    raw_coords = record["frac_coords"]
    if len(raw_coords) != len(atomic_numbers):
        raise ValueError("frac_coords length does not match atomic_numbers")
    frac_coords = [_finite_numbers(row, "frac_coords") for row in raw_coords]
    if any(len(row) != 3 for row in frac_coords):
        raise ValueError("each fractional coordinate must have length 3")

    raw_lattice = record["lattice_matrix"]
    if len(raw_lattice) != 3:
        raise ValueError("lattice_matrix must have shape [3, 3]")
    lattice = [_finite_numbers(row, "lattice_matrix") for row in raw_lattice]
    if any(len(row) != 3 for row in lattice):
        raise ValueError("lattice_matrix must have shape [3, 3]")
    return atomic_numbers, frac_coords, lattice


def record_to_structure(record: dict[str, Any]):
    """Build a pymatgen Structure without applying a validity pre-filter."""

    try:
        from pymatgen.core import Lattice, Structure
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("CIF export requires: pip install -e '.[mp20]'") from exc

    atomic_numbers, frac_coords, lattice_matrix = validate_generated_record(record)
    lattice = Lattice(lattice_matrix)
    if not math.isfinite(float(lattice.volume)) or float(lattice.volume) <= 1.0e-8:
        raise ValueError("lattice has zero or non-finite volume")
    return Structure(
        lattice,
        atomic_numbers,
        frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _prepare_cif_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_cifs = list(output_dir.glob("*.cif"))
    if old_cifs and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains CIFs; choose a new directory "
            "or pass --overwrite"
        )
    if overwrite:
        for path in old_cifs:
            path.unlink()


def export_generated_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export every parseable generated record and retain failures in a manifest."""

    try:
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("CIF export requires: pip install -e '.[mp20]'") from exc

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"generated JSONL not found: {input_path}")
    _prepare_cif_directory(output_dir, overwrite)

    manifest_path = output_dir / "manifest.jsonl"
    total = exported = failed = 0
    with input_path.open(encoding="utf-8") as source, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            total += 1
            sample_id = f"sample_{index:06d}"
            entry: dict[str, Any] = {"sample_id": sample_id, "source_line": index + 1}
            try:
                record = json.loads(line)
                structure = record_to_structure(record)
                filename = f"{sample_id}.cif"
                CifWriter(structure, symprec=None).write_file(str(output_dir / filename))
                entry.update(
                    {
                        "status": "exported",
                        "cif": filename,
                        "formula": structure.composition.reduced_formula,
                        "num_atoms": len(structure),
                        "generated_formation_energy_per_atom": record.get(
                            "generated_formation_energy_per_atom"
                        ),
                        "energy_is_independently_verified": bool(
                            record.get("energy_is_independently_verified", False)
                        ),
                    }
                )
                exported += 1
            except Exception as exc:
                entry.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failed += 1
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path.resolve()),
        "cif_directory": str(output_dir.resolve()),
        "generated_records": total,
        "exported_cifs": exported,
        "failed_exports": failed,
        "cif_export_rate": exported / total if total else 0.0,
        "manifest": str(manifest_path.resolve()),
        "benchmark_note": "Run LeMat-GenBench on this unfiltered CIF directory.",
    }
    with (output_dir / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
