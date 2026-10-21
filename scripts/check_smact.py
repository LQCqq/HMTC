#!/usr/bin/env python3
"""Annotate CIFs with SMACT and optionally copy valid structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hmtc.evaluation.smact import annotate_cif_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-cif-dir", type=Path)
    parser.add_argument("--oxidation-states-set", default="smact14")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = annotate_cif_directory(
        args.cifs,
        args.output,
        oxidation_states_set=args.oxidation_states_set,
        valid_cif_dir=args.valid_cif_dir,
        overwrite_valid_cif_dir=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
