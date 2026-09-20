import csv

import pytest

from hmtc.data.mp20 import discover_mp20_splits


def test_discover_mp20_preserves_named_splits(tmp_path) -> None:
    for split in ("train", "val", "test"):
        with (tmp_path / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["material_id", "cif", "formation_energy_per_atom"])
            writer.writeheader()
    splits = discover_mp20_splits(tmp_path)
    assert set(splits) == {"train", "val", "test"}


def test_discover_mp20_requires_train(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_mp20_splits(tmp_path)
