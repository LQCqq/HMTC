"""MACE-MP relaxation adapter for generated CIF structures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MACE_MODEL = "medium-0b3"


def create_mace_calculator(
    *,
    model: str = DEFAULT_MACE_MODEL,
    device: str = "cuda",
    default_dtype: str = "float32",
):
    """Create the MACE-MP calculator used for energies and relaxation."""

    try:
        from mace.calculators import mace_mp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MACE relaxation requires: pip install -e '.[mace]'") from exc
    return mace_mp(model=model, device=device, default_dtype=default_dtype)


def relax_structure(
    structure,
    calculator,
    *,
    fmax: float = 0.02,
    steps: int = 50,
):
    """Relax atomic positions and cell using LeMat-GenBench-compatible settings."""

    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGSLineSearch
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    optimizer = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_force = max(
        (math.sqrt(float((force * force).sum())) for force in forces), default=0.0
    )
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    metadata = {
        "converged": converged,
        "steps": int(optimizer.get_number_of_steps()),
        "energy_eV": energy,
        "energy_per_atom_eV": energy / len(atoms),
        "max_force_eV_per_A": max_force,
    }
    return relaxed, metadata


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
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


def _supported_atomic_numbers(calculator) -> set[int] | None:
    """Read the element coverage exposed by different MACE calculator versions."""

    values = getattr(calculator, "atomic_numbers", None)
    if values is None:
        z_table = getattr(calculator, "z_table", None)
        values = getattr(z_table, "zs", None)
    if values is None:
        models = getattr(calculator, "models", None)
        if models:
            values = getattr(models[0], "atomic_numbers", None)
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {int(value) for value in values}


def relax_cif_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    model: str = DEFAULT_MACE_MODEL,
    device: str = "cuda",
    default_dtype: str = "float32",
    fmax: float = 0.02,
    steps: int = 50,
    start_index: int = 0,
    max_structures: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Relax CIFs with one reused MACE calculator and write an audit manifest."""

    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MACE relaxation requires: pip install -e '.[mace]'") from exc

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_structures is not None and max_structures <= 0:
        raise ValueError("max_structures must be positive")

    all_cif_paths = sorted(input_dir.rglob("*.cif"))
    cif_paths = all_cif_paths[start_index:]
    if max_structures is not None:
        cif_paths = cif_paths[:max_structures]
    if not cif_paths:
        raise ValueError(
            f"no CIF files selected from {input_dir} with start_index={start_index}"
        )
    _prepare_output_directory(output_dir, overwrite)
    calculator = create_mace_calculator(
        model=model, device=device, default_dtype=default_dtype
    )
    supported_atomic_numbers = _supported_atomic_numbers(calculator)

    succeeded = failed = converged_count = unsupported_count = 0
    manifest_path = output_dir / "mace_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for local_index, cif_path in enumerate(cif_paths):
            source_index = start_index + local_index
            entry: dict[str, Any] = {
                "source_index": source_index,
                "source_cif": str(cif_path.resolve()),
            }
            try:
                structure = Structure.from_file(cif_path)
                atomic_numbers = {int(number) for number in structure.atomic_numbers}
                unsupported = (
                    sorted(atomic_numbers - supported_atomic_numbers)
                    if supported_atomic_numbers is not None
                    else []
                )
                if unsupported:
                    entry.update(
                        {
                            "status": "unsupported_elements",
                            "unsupported_atomic_numbers": unsupported,
                            "error": "structure contains elements not supported by the MACE model",
                        }
                    )
                    unsupported_count += 1
                    failed += 1
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed = local_index + 1
                    if completed % 10 == 0 or completed == len(cif_paths):
                        print(
                            f"processed {completed}/{len(cif_paths)} CIFs "
                            f"(relaxed={succeeded}, failed={failed})",
                            flush=True,
                        )
                    continue
                relaxed, metadata = relax_structure(
                    structure, calculator, fmax=fmax, steps=steps
                )
                filename = f"relaxed_{source_index:06d}_{cif_path.name}"
                CifWriter(relaxed, symprec=None).write_file(str(output_dir / filename))
                entry.update({"status": "relaxed", "relaxed_cif": filename, **metadata})
                succeeded += 1
                converged_count += int(metadata["converged"])
            except Exception as exc:
                entry.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failed += 1
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            completed = local_index + 1
            if completed % 10 == 0 or completed == len(cif_paths):
                print(
                    f"processed {completed}/{len(cif_paths)} CIFs "
                    f"(relaxed={succeeded}, failed={failed})",
                    flush=True,
                )

    total = len(cif_paths)
    summary = {
        "input_cif_directory": str(input_dir.resolve()),
        "output_cif_directory": str(output_dir.resolve()),
        "total_cifs": total,
        "available_cifs": len(all_cif_paths),
        "start_index": start_index,
        "relaxed": succeeded,
        "failed": failed,
        "unsupported_elements": unsupported_count,
        "converged": converged_count,
        "relaxation_success_rate": succeeded / total,
        "convergence_rate": converged_count / total,
        "model": model,
        "device": device,
        "default_dtype": default_dtype,
        "fmax": fmax,
        "steps": steps,
        "supported_atomic_numbers": (
            sorted(supported_atomic_numbers)
            if supported_atomic_numbers is not None
            else None
        ),
        "manifest": str(manifest_path.resolve()),
    }
    with Path("/dev/null").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
