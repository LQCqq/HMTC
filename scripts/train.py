#!/usr/bin/env python3
"""Minimal real-data trainer for preprocessed JSONL crystal records."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations
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
    final_train_loss = 0.0
    final_train_terms: dict[str, float] = {}
    final_weighted_train_terms: dict[str, float] = {}
    final_train_diagnostics: dict[str, float | list[float]] = {}

    for epoch in range(1, args.epochs + 1):
        running = torch.zeros((), device=args.device)
        term_sums: dict[str, torch.Tensor] = {}
        diagnostic_sums: dict[str, torch.Tensor] = {}
        diagnostic_samples = 0
        count = 0
        collect_diagnostics = epoch == args.epochs

        for state in loader:
            state = state.to(args.device)
            result = trainer.train_step(
                state,
                collect_diagnostics=collect_diagnostics,
            )
            running += result.loss
            for name, value in result.terms.items():
                if name not in term_sums:
                    term_sums[name] = torch.zeros_like(value)
                term_sums[name] += value
            if collect_diagnostics:
                batch_size = state.batch_size
                for name, value in result.diagnostics.items():
                    if name not in diagnostic_sums:
                        diagnostic_sums[name] = torch.zeros_like(value)
                    diagnostic_sums[name] += value * batch_size
                diagnostic_samples += batch_size
            count += 1

        denominator = max(count, 1)
        epoch_loss = float((running / denominator).cpu())
        epoch_terms = {
            name: float((value / denominator).cpu()) for name, value in term_sums.items()
        }
        weighted_epoch_terms = {
            name: value * getattr(loss_config, name) for name, value in epoch_terms.items()
        }
        print(f"epoch={epoch:04d} loss={epoch_loss:.6f}", flush=True)

        if epoch == args.epochs:
            final_train_loss = epoch_loss
            final_train_terms = epoch_terms
            final_weighted_train_terms = weighted_epoch_terms
            diagnostic_denominator = max(diagnostic_samples, 1)
            for name, value in diagnostic_sums.items():
                averaged = (value / diagnostic_denominator).cpu()
                if averaged.numel() == 1:
                    final_train_diagnostics[name] = float(averaged)
                else:
                    final_train_diagnostics[name] = averaged.tolist()

    print("=" * 72)
    print("Final training loss breakdown")
    print("=" * 72)
    print(f"{'term':<28} {'raw':>12} {'weight':>10} {'weighted':>12}")
    for name, value in final_train_terms.items():
        weight = getattr(loss_config, name)
        weighted = final_weighted_train_terms[name]
        print(f"{name:<28} {value:>12.6f} {weight:>10.6f} {weighted:>12.6f}")
    print("-" * 72)
    weighted_sum = sum(final_weighted_train_terms.values())
    print(f"{'weighted_sum':<28} {'':>12} {'':>10} {weighted_sum:>12.6f}")
    print(f"{'total':<28} {final_train_loss:>12.6f}", flush=True)

    print("=" * 72)
    print("Final routing diagnostics")
    print("=" * 72)
    tree_names = [spec.name for spec in model_config.tree_specs]
    for spec in model_config.tree_specs:
        for level_index in range(1, len(spec.groups) + 1):
            prefix = f"routing.{spec.name}.level{level_index}"
            occupancy = final_train_diagnostics.get(f"{prefix}.occupancy", [])
            entropy = final_train_diagnostics.get(f"{prefix}.entropy", float("nan"))
            occupancy_text = ", ".join(f"{value:.4f}" for value in occupancy)
            print(
                f"{spec.name:<10} level={level_index} "
                f"entropy={entropy:.6f} occupancy=[{occupancy_text}]"
            )
    for left_name, right_name in combinations(tree_names, 2):
        left_spec = next(spec for spec in model_config.tree_specs if spec.name == left_name)
        right_spec = next(spec for spec in model_config.tree_specs if spec.name == right_name)
        for level_index in range(
            1, min(len(left_spec.groups), len(right_spec.groups)) + 1
        ):
            key = f"routing_similarity.{left_name}__{right_name}.level{level_index}"
            if key in final_train_diagnostics:
                print(
                    f"offdiag_similarity {left_name}<->{right_name} level={level_index} "
                    f"value={final_train_diagnostics[key]:.6f}"
                )
    fusion_text = " ".join(
        f"{name}={final_train_diagnostics.get(f'fusion.{name}', float('nan')):.6f}"
        for name in tree_names
    )
    print(f"fusion {fusion_text}", flush=True)

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
            "final_train_loss": final_train_loss,
            "final_train_terms": final_train_terms,
            "final_weighted_train_terms": final_weighted_train_terms,
            "final_train_diagnostics": final_train_diagnostics,
        },
        args.output,
    )
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
