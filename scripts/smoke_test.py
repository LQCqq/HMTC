#!/usr/bin/env python3
"""One forward/backward step plus two-step sampling on synthetic tensors."""

from __future__ import annotations

from pathlib import Path

import torch

from hmtc.config import load_config
from hmtc.data import NormalizationStats, synthetic_batch
from hmtc.diffusion import HMTFSampler
from hmtc.models import HMTFModel
from hmtc.training import HMTFTrainer


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    model_config, diffusion_config, loss_config, _ = load_config(project / "conf/base.yaml")
    stats = NormalizationStats(0.0, 1.0,
                               (1.5, 0.0, 1.5, 0.0, 0.0, 1.5), (0.2,) * 6)
    model = HMTFModel(model_config, diffusion_config, stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    trainer = HMTFTrainer(model, optimizer, diffusion_config, loss_config)
    clean = synthetic_batch(2, model_config.max_sites, model_config.num_elements)
    result = trainer.train_step(clean)
    print(f"train loss: {result.loss.item():.6f}")
    model.eval()
    sampler = HMTFSampler(model, model_config, diffusion_config)
    counts = torch.tensor([3, min(12, model_config.max_sites)])
    sampled = sampler.sample(batch_size=2, steps=2, num_atoms=counts)
    print("sample shapes:", sampled.energy.shape, sampled.elements.shape, sampled.frac_coords.shape)
    print("active sites:", sampled.existence.sum(dim=1).tolist())
    assert torch.equal(sampled.num_atoms.cpu(), counts)
    assert torch.isfinite(sampled.frac_coords).all()
    print("fixed-count sampling OK")


if __name__ == "__main__":
    main()
