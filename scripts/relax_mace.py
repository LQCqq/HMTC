#!/usr/bin/env python3
"""Relax generated CIFs with MACE-MP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hmtc.evaluation.mace import DEFAULT_MACE_MODEL, relax_cif_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MACE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = relax_cif_directory(
        args.cifs,
        args.output_dir,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        fmax=args.fmax,
        steps=args.steps,
        start_index=args.start_index,
        max_structures=args.max_structures,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
