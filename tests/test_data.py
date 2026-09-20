from pathlib import Path

import torch

from hmtc.data import JsonlCrystalDataset, collate_records, compute_energy_stats


def test_jsonl_data_contract() -> None:
    path = Path(__file__).parent / "fixtures/tiny.jsonl"
    dataset = JsonlCrystalDataset(path)
    stats = compute_energy_stats(dataset)
    state = collate_records([dataset[0], dataset[1]], 100, 20, stats)
    assert state.energy.shape == (2, 1)
    assert torch.allclose(state.energy.mean(), torch.tensor(0.0), atol=1.0e-6)
    assert state.existence.sum().item() == 6
    assert state.elements[0, 0].item() == 2
