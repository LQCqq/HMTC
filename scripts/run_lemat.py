#!/usr/bin/env python3
"""Run an external LeMat-GenBench checkout on raw HMTC CIFs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from hmtc.evaluation.lemat import build_lemat_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemat-dir", type=Path, required=True)
    parser.add_argument("--cifs", type=Path, required=True)
    parser.add_argument("--config", default="comprehensive")
    parser.add_argument("--name", required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lemat_dir = args.lemat_dir.resolve()
    cifs = args.cifs.resolve()
    if not (lemat_dir / "scripts" / "run_benchmarks.py").is_file():
        raise FileNotFoundError(f"not a LeMat-GenBench checkout: {lemat_dir}")
    if not cifs.is_dir():
        raise FileNotFoundError(f"CIF directory not found: {cifs}")
    if not any(cifs.rglob("*.cif")):
        raise ValueError(f"no CIF files found in: {cifs}")
    command = build_lemat_command(
        cifs=cifs,
        config=args.config,
        name=args.name,
        uv=args.uv,
        families=args.families,
        batch_size=args.batch_size,
    )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=lemat_dir, check=True)


if __name__ == "__main__":
    main()
