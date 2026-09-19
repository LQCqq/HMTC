#!/usr/bin/env python3
"""Minimal real-data trainer for preprocessed JSONL crystal records."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hmtc.config import load_config
from hmtc.data import JsonlCrystalDataset, collate_records, compute_energy_stats
from hmtc.models import HMTFModel
from hmtc.training import HMTFTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", type=Path, help="preprocessed training JSONL")
    data_group.add_argument(
        "--data-dir", type=Path, help="MP-20 directory containing train.jsonl"
    )
    parser.add_argument("--config", type=Path, default=Path("conf/base.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model_config, diffusion_config, loss_config, _ = load_config(args.config)
    data_path = args.data if args.data is not None else args.data_dir / "train.jsonl"
    dataset = JsonlCrystalDataset(data_path)
    stats = compute_energy_stats(dataset)

    def collate(batch: list) -> object:
        return collate_records(
            batch,
            num_elements=model_config.num_elements,
            max_sites=model_config.max_sites,
            stats=stats,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    model = HMTFModel(model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    trainer = HMTFTrainer(model, optimizer, diffusion_config, loss_config)
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for state in loader:
            result = trainer.train_step(state.to(args.device))
            running += float(result.loss)
            count += 1
        print(f"epoch={epoch:04d} loss={running / max(count, 1):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model_config),
            "diffusion_config": asdict(diffusion_config),
            "loss_config": asdict(loss_config),
            "normalization": asdict(stats),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epochs": args.epochs,
        },
        args.output,
    )
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
