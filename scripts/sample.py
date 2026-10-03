#!/usr/bin/env python3
"""Generate JSONL crystal records and CIF files from an HMTF checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=None,
        help="CIF output directory; defaults to <output_stem>_cifs beside the JSONL",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
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
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        raise ImportError(
            "CIF export requires pymatgen; install with "
            "`python -m pip install -e '.[evaluation]'`"
        ) from exc

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_config = model_config_from_dict(checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    stats = NormalizationStats(**checkpoint["normalization"])
    model = HMTFModel(model_config).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sampler = HMTFSampler(model, model_config, diffusion_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cif_dir = args.cif_dir
    if cif_dir is None:
        cif_dir = args.output.parent / f"{args.output.stem}_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.parent / f"{args.output.stem}_cif_summary.json"
    cifs_written = 0
    failure_counts: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, args.num_samples, args.batch_size):
            current_batch_size = min(args.batch_size, args.num_samples - start)
            state = sampler.sample(
                batch_size=current_batch_size,
                steps=args.steps,
                device=args.device,
                temperature=args.temperature,
                energy_clamp=args.energy_clamp,
            )
            lattice = lattice_params_to_matrix(state.lattice).cpu()
            energy = stats.denormalize_energy(state.energy).cpu()
            state = state.to("cpu")

            for local_index in range(current_batch_size):
                sample_index = start + local_index
                sample_id = f"sample_{sample_index:06d}"
                active = state.existence[local_index].bool()
                atomic_numbers = (state.elements[local_index, active] + 1).tolist()
                frac_coords = state.frac_coords[local_index, active]
                lattice_matrix = lattice[local_index]
                cif_path = cif_dir / f"{sample_id}.cif"
                record = {
                    "sample_id": sample_id,
                    "generated_formation_energy_per_atom": float(
                        energy[local_index, 0]
                    ),
                    "atomic_numbers": atomic_numbers,
                    "frac_coords": frac_coords.tolist(),
                    "lattice_matrix": lattice_matrix.tolist(),
                    "generated_composition_factor": state.composition[
                        local_index
                    ].tolist(),
                    "energy_is_independently_verified": False,
                }

                try:
                    if not atomic_numbers:
                        raise ValueError("empty_structure")
                    if not torch.isfinite(frac_coords).all():
                        raise ValueError("non_finite_coordinates")
                    if not torch.isfinite(lattice_matrix).all():
                        raise ValueError("non_finite_lattice")
                    volume = float(torch.linalg.det(lattice_matrix).abs())
                    if volume <= 1.0e-6:
                        raise ValueError("singular_lattice")
                    structure = Structure(
                        Lattice(lattice_matrix.tolist()),
                        atomic_numbers,
                        frac_coords.tolist(),
                        coords_are_cartesian=False,
                        to_unit_cell=True,
                    )
                    CifWriter(structure).write_file(cif_path)
                    record["cif_written"] = True
                    record["cif_file"] = cif_path.name
                    cifs_written += 1
                except (ArithmeticError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                    reason = str(exc) or type(exc).__name__
                    record["cif_written"] = False
                    record["cif_error"] = reason
                    failure_counts[reason] += 1

                handle.write(json.dumps(record) + "\n")

            completed = start + current_batch_size
            print(f"sampled {completed}/{args.num_samples}", flush=True)

    summary = {
        "requested_samples": args.num_samples,
        "jsonl_records": args.num_samples,
        "cifs_written": cifs_written,
        "cif_failures": args.num_samples - cifs_written,
        "failure_counts": dict(sorted(failure_counts.items())),
        "jsonl": str(args.output),
        "cif_dir": str(cif_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote JSONL records: {args.output}")
    print(f"wrote CIF files: {cifs_written}/{args.num_samples} -> {cif_dir}")
    print(f"wrote CIF summary: {summary_path}")


if __name__ == "__main__":
    main()
