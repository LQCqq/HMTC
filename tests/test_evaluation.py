import json
from pathlib import Path

import pytest

from hmtc.evaluation.io import export_generated_jsonl, validate_generated_record
from hmtc.evaluation.lemat import build_lemat_command
from hmtc.evaluation.mace import relax_structure
from hmtc.evaluation.smact import check_formula, copy_smact_valid_cifs


def test_validate_generated_record() -> None:
    numbers, coords, lattice = validate_generated_record(
        {
            "atomic_numbers": [14, 8, 8],
            "frac_coords": [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            "lattice_matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        }
    )
    assert numbers == [14, 8, 8]
    assert len(coords) == 3
    assert lattice[0] == [4.0, 0.0, 0.0]


def test_validate_generated_record_rejects_empty_structure() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_generated_record(
            {
                "atomic_numbers": [],
                "frac_coords": [],
                "lattice_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        )


def test_export_preserves_failed_records_in_denominator(tmp_path: Path) -> None:
    pytest.importorskip("pymatgen")
    generated = tmp_path / "generated.jsonl"
    records = [
        {
            "atomic_numbers": [14, 8, 8],
            "frac_coords": [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            "lattice_matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        },
        {
            "atomic_numbers": [],
            "frac_coords": [],
            "lattice_matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        },
    ]
    generated.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "raw_cifs"
    summary = export_generated_jsonl(generated, output)
    assert summary["generated_records"] == 2
    assert summary["exported_cifs"] == 1
    assert summary["failed_exports"] == 1
    assert len(list(output.glob("sample_*.cif"))) == 1


def test_relaxation_adapter_with_ase_calculator() -> None:
    pytest.importorskip("ase")
    pymatgen = pytest.importorskip("pymatgen.core")
    from ase.calculators.emt import EMT

    structure = pymatgen.Structure(
        pymatgen.Lattice.cubic(3.6), [29], [[0.0, 0.0, 0.0]]
    )
    relaxed, metadata = relax_structure(structure, EMT(), fmax=10.0, steps=1)
    assert len(relaxed) == 1
    assert metadata["steps"] <= 1
    assert metadata["energy_per_atom_eV"] == pytest.approx(metadata["energy_eV"])


def test_smact_uses_lemat_oxidation_set() -> None:
    calls = []

    def checker(formula: str, **kwargs) -> bool:
        calls.append((formula, kwargs))
        return True

    assert check_formula("Na Cl", checker=checker)
    assert calls == [("NaCl", {"oxidation_states_set": "smact14"})]


def test_copy_smact_valid_cifs_keeps_only_valid_records(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    valid_cif = source_dir / "valid.cif"
    invalid_cif = source_dir / "invalid.cif"
    valid_cif.write_text("valid", encoding="utf-8")
    invalid_cif.write_text("invalid", encoding="utf-8")
    results = tmp_path / "smact.jsonl"
    records = [
        {
            "cif": str(valid_cif),
            "status": "ok",
            "formula": "NaCl",
            "smact_valid": True,
        },
        {
            "cif": str(invalid_cif),
            "status": "ok",
            "formula": "Na2Cl",
            "smact_valid": False,
        },
    ]
    results.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    output = tmp_path / "filtered"
    summary = copy_smact_valid_cifs(results, output)
    assert summary["smact_valid_records"] == 1
    assert summary["copied_cifs"] == 1
    assert summary["failed_copies"] == 0
    assert (output / "valid.cif").read_text(encoding="utf-8") == "valid"
    assert not (output / "invalid.cif").exists()


def test_smact_annotation_can_copy_valid_cifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pymatgen = pytest.importorskip("pymatgen.core")
    from hmtc.evaluation import smact as smact_module

    raw = tmp_path / "raw"
    raw.mkdir()
    structure = pymatgen.Structure(
        pymatgen.Lattice.cubic(5.6), [11, 17], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    structure.to(filename=raw / "nacl.cif")
    monkeypatch.setattr(smact_module, "check_formula", lambda *args, **kwargs: True)

    valid_dir = tmp_path / "smact_structure"
    summary = smact_module.annotate_cif_directory(
        raw,
        tmp_path / "results.jsonl",
        valid_cif_dir=valid_dir,
    )
    assert summary["smact_valid"] == 1
    assert summary["copied_valid_cifs"] == 1
    assert summary["failed_valid_cif_copies"] == 0
    assert (valid_dir / "nacl.cif").is_file()


def test_build_lemat_command() -> None:
    cifs = Path("generated/raw_cifs")
    command = build_lemat_command(
        cifs=cifs,
        config="comprehensive",
        name="hmtc_mp20",
        families=["distribution", "stability"],
        batch_size=64,
    )
    assert command[:3] == ["uv", "run", "scripts/run_benchmarks.py"]
    assert command[command.index("--cifs") + 1] == str(cifs.resolve())
    assert command[-5:] == [
        "--families",
        "distribution",
        "stability",
        "--batch-size",
        "64",
    ]
