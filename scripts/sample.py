#!/usr/bin/env python3
"""Generate crystal-factor records from an HMTF checkpoint without CFG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hmtc.config import DiffusionConfig, HMTFConfig, TreeSpec
from hmtc.data import NormalizationStats
from hmtc.diffusion import HMTFSampler
from hmtc.models import HMTFModel
from hmtc.models.geometry import lattice_params_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--energy-clamp", type=float, default=None, help="optional normalized value"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def model_config_from_dict(raw: dict) -> HMTFConfig:
    item = dict(raw)
    item["tree_specs"] = tuple(
        TreeSpec(**{**spec, "groups": tuple(spec["groups"])}) for spec in item["tree_specs"]
    )
    return HMTFConfig(**item)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_config = model_config_from_dict(checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    stats = NormalizationStats(**checkpoint["normalization"])
    model = HMTFModel(model_config).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sampler = HMTFSampler(model, model_config, diffusion_config)
    state = sampler.sample(
        batch_size=args.num_samples,
        steps=args.steps,
        device=args.device,
        temperature=args.temperature,
        energy_clamp=args.energy_clamp,
    )
    lattice = lattice_params_to_matrix(state.lattice).cpu()
    energy = stats.denormalize_energy(state.energy).cpu()
    state = state.to("cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.num_samples):
            active = state.existence[index].bool()
            record = {
                "generated_formation_energy_per_atom": float(energy[index, 0]),
                "atomic_numbers": (state.elements[index, active] + 1).tolist(),
                "frac_coords": state.frac_coords[index, active].tolist(),
                "lattice_matrix": lattice[index].tolist(),
                "generated_composition_factor": state.composition[index].tolist(),
                "energy_is_independently_verified": False,
            }
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {args.num_samples} generated records: {args.output}")


if __name__ == "__main__":
    main()
