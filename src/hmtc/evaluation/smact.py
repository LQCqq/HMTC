"""SMACT composition-validity diagnostics for generated CIFs."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import Any, Callable


def check_formula(
    formula: str,
    *,
    oxidation_states_set: str = "smact14",
    checker: Callable[..., bool] | None = None,
) -> bool:
    """Evaluate one formula with the same SMACT setting used by LeMat-GenBench."""

    if checker is None:
        try:
            from smact.screening import smact_validity
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("SMACT evaluation requires: pip install -e '.[evaluation]'") from exc
        checker = smact_validity
    return bool(checker(formula.replace(" ", ""), oxidation_states_set=oxidation_states_set))


def annotate_cif_directory(
    cif_dir: Path,
    output_path: Path,
    *,
    oxidation_states_set: str = "smact14",
    valid_cif_dir: Path | None = None,
    overwrite_valid_cif_dir: bool = False,
) -> dict[str, Any]:
    """Annotate every CIF and optionally copy valid structures to a new directory."""

    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("SMACT evaluation requires: pip install -e '.[evaluation]'") from exc

    cif_dir = Path(cif_dir)
    output_path = Path(output_path)
    cif_paths = sorted(cif_dir.rglob("*.cif"))
    if not cif_paths:
        raise ValueError(f"no CIF files found in {cif_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_dir: Path | None = None
    if valid_cif_dir is not None:
        filtered_dir = Path(valid_cif_dir)
        input_resolved = cif_dir.resolve()
        filtered_resolved = filtered_dir.resolve()
        output_is_nested = input_resolved in filtered_resolved.parents
        if filtered_resolved == input_resolved or output_is_nested:
            raise ValueError("valid_cif_dir must not be inside the input CIF directory")
        _prepare_cif_output_directory(filtered_dir, overwrite_valid_cif_dir)

    valid = invalid = errors = 0
    copied_valid = failed_valid_copies = 0
    formula_cache: dict[str, bool] = {}
    copied_destinations: set[Path] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, cif_path in enumerate(cif_paths, start=1):
            entry: dict[str, Any] = {"cif": str(cif_path.resolve())}
            try:
                structure = Structure.from_file(cif_path)
                formula = structure.composition.reduced_formula.replace(" ", "")
                if formula not in formula_cache:
                    formula_cache[formula] = check_formula(
                        formula,
                        oxidation_states_set=oxidation_states_set,
                    )
                result = formula_cache[formula]
                entry.update(
                    {"status": "ok", "formula": formula, "smact_valid": result}
                )
                valid += int(result)
                invalid += int(not result)
                if result and filtered_dir is not None:
                    destination = filtered_dir / cif_path.name
                    try:
                        if destination in copied_destinations:
                            raise FileExistsError(
                                f"duplicate CIF filename: {cif_path.name}"
                            )
                        copy2(cif_path, destination)
                        copied_destinations.add(destination)
                        entry["filtered_cif"] = str(destination.resolve())
                        copied_valid += 1
                    except Exception as copy_exc:
                        entry.update(
                            {
                                "copy_error_type": type(copy_exc).__name__,
                                "copy_error": str(copy_exc),
                            }
                        )
                        failed_valid_copies += 1
            except Exception as exc:
                entry.update(
                    {
                        "status": "error",
                        "smact_valid": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                errors += 1
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(cif_paths):
                print(
                    f"checked {index}/{len(cif_paths)} CIFs "
                    f"({len(formula_cache)} unique formulas)",
                    flush=True,
                )

    total = len(cif_paths)
    summary = {
        "cif_directory": str(cif_dir.resolve()),
        "total_cifs": total,
        "smact_valid": valid,
        "smact_invalid": invalid,
        "errors": errors,
        "unique_formulas": len(formula_cache),
        "smact_valid_rate": valid / total,
        "oxidation_states_set": oxidation_states_set,
        "results": str(output_path.resolve()),
        "valid_cif_directory": (
            str(filtered_dir.resolve()) if filtered_dir is not None else None
        ),
        "copied_valid_cifs": copied_valid,
        "failed_valid_cif_copies": failed_valid_copies,
        "benchmark_note": "Diagnostic only; do not pre-filter LeMat-GenBench input.",
    }
    summary_path = Path("/dev/null")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def _prepare_cif_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_cifs = list(output_dir.glob("*.cif"))
    if old_cifs and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains CIFs; choose a new directory "
            "or enable overwrite"
        )
    if overwrite:
        for path in old_cifs:
            path.unlink()


def copy_smact_valid_cifs(
    results_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy CIFs marked valid by a previous SMACT diagnostic run.

    The original CIF directory is never modified.  The input is the JSONL file
    written by :func:`annotate_cif_directory`.
    """

    results_path = Path(results_path)
    output_dir = Path(output_dir)
    if not results_path.is_file():
        raise FileNotFoundError(f"SMACT results not found: {results_path}")

    _prepare_cif_output_directory(output_dir, overwrite)

    records = valid_records = copied = failed = 0
    manifest_path = output_dir / "smact_filter_manifest.jsonl"
    seen_destinations: set[Path] = set()
    with results_path.open(encoding="utf-8") as source_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            records += 1
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {results_path}"
                ) from exc

            if result.get("status") != "ok" or result.get("smact_valid") is not True:
                continue

            valid_records += 1
            source = Path(result["cif"])
            destination = output_dir / source.name
            entry: dict[str, Any] = {
                "source_cif": str(source),
                "formula": result.get("formula"),
                "smact_valid": True,
            }
            try:
                if not source.is_file():
                    raise FileNotFoundError(f"source CIF not found: {source}")
                if destination in seen_destinations:
                    raise FileExistsError(
                        f"duplicate CIF filename in SMACT results: {source.name}"
                    )
                copy2(source, destination)
                seen_destinations.add(destination)
                entry.update(
                    {
                        "status": "copied",
                        "filtered_cif": str(destination.resolve()),
                    }
                )
                copied += 1
            except Exception as exc:
                entry.update(
                    {
                        "status": "copy_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failed += 1
            manifest_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "smact_results": str(results_path.resolve()),
        "output_cif_directory": str(output_dir.resolve()),
        "result_records": records,
        "smact_valid_records": valid_records,
        "copied_cifs": copied,
        "failed_copies": failed,
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = output_dir / "smact_filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
