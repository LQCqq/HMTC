#!/usr/bin/env python3
"""Inspect learned tree routing and fusion from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hmtc.config import DiffusionConfig, HMTFConfig, TreeSpec
from hmtc.data import JsonlCrystalDataset, NormalizationStats, collate_records
from hmtc.diagnostics import routing_diagnostics
from hmtc.diffusion import JointDiffusion
from hmtc.models import HMTFModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--times", type=float, nargs="+", default=(0.1, 0.5, 0.9))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def model_config_from_dict(raw: dict) -> HMTFConfig:
    item = dict(raw)
    item["tree_specs"] = tuple(
        TreeSpec(**{**spec, "groups": tuple(spec["groups"])})
        for spec in item["tree_specs"]
    )
    return HMTFConfig(**item)


def print_diagnostics(
    values: dict[str, float | list[float]],
    model_config: HMTFConfig,
) -> None:
    tree_specs = [spec for spec in model_config.tree_specs if spec.leaf_entity == "site"]
    tree_names = [spec.name for spec in tree_specs]
    for spec in tree_specs:
        for level_index in range(1, len(spec.groups) + 1):
            prefix = f"routing.{spec.name}.level{level_index}"
            occupancy = values.get(f"{prefix}.occupancy", [])
            entropy = values.get(f"{prefix}.entropy", float("nan"))
            occupancy_text = ", ".join(f"{value:.4f}" for value in occupancy)
            print(
                f"{spec.name:<10} level={level_index} "
                f"entropy={entropy:.6f} occupancy=[{occupancy_text}]"
            )
    for left_name, right_name in combinations(tree_names, 2):
        left_spec = next(spec for spec in tree_specs if spec.name == left_name)
        right_spec = next(spec for spec in tree_specs if spec.name == right_name)
        for level_index in range(
            1, min(len(left_spec.groups), len(right_spec.groups)) + 1
        ):
            key = f"routing_similarity.{left_name}__{right_name}.level{level_index}"
            if key in values:
                print(
                    f"offdiag_similarity {left_name}<->{right_name} level={level_index} "
                    f"value={values[key]:.6f}"
                )
    fusion_text = " ".join(
        f"{name}={values.get(f'fusion.{name}', float('nan')):.6f}"
        for name in tree_names
    )
    print(f"fusion {fusion_text}")


def main() -> None:
    args = parse_args()
    if any(time < 0.0 or time > 1.0 for time in args.times):
        raise ValueError("every --times value must lie in [0, 1]")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_config = model_config_from_dict(checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    stats = NormalizationStats(**checkpoint["normalization"])
    dataset = JsonlCrystalDataset(args.data_dir / f"{args.split}.jsonl")

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
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )
    model = HMTFModel(model_config).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    diffusion = JointDiffusion(diffusion_config, model_config.num_elements)
    results: dict[str, dict[str, object]] = {}

    with torch.inference_mode():
        for noise_time in args.times:
            sums: dict[str, torch.Tensor] = {}
            sample_count = 0
            for batch_index, clean in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                clean = clean.to(args.device)
                time = torch.full(
                    (clean.batch_size,),
                    noise_time,
                    device=args.device,
                    dtype=clean.energy.dtype,
                )
                noisy, _ = diffusion.q_sample(clean, time)
                output = model(noisy, time)
                active = clean.resolved_site_mask() & (clean.existence > 0.5)
                batch_values = routing_diagnostics(output, active)
                for name, value in batch_values.items():
                    if name not in sums:
                        sums[name] = torch.zeros_like(value)
                    sums[name] += value * clean.batch_size
                sample_count += clean.batch_size

            denominator = max(sample_count, 1)
            values: dict[str, float | list[float]] = {}
            for name, value in sums.items():
                averaged = (value / denominator).cpu()
                values[name] = (
                    float(averaged) if averaged.numel() == 1 else averaged.tolist()
                )
            key = f"t={noise_time:.3f}"
            results[key] = {"samples": sample_count, "diagnostics": values}
            print("=" * 88)
            print(f"split={args.split} noise_time={noise_time:.3f} samples={sample_count}")
            print("=" * 88)
            print_diagnostics(values, model_config)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint.resolve()),
            "split": args.split,
            "seed": args.seed,
            "results": results,
        }
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"saved diagnostics: {args.output}")


if __name__ == "__main__":
    main()
