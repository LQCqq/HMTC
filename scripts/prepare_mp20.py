#!/usr/bin/env python3
"""Convert standard CDVAE MP-20 CSV splits into HMTF JSONL splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hmtc.data import convert_mp20_csv, discover_mp20_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--num-elements", type=int, default=100)
    args = parser.parse_args()
    sources = discover_mp20_splits(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for split, source in sources.items():
        if source.suffix == ".jsonl":
            raise ValueError(
                f"{source} is already JSONL; pass its directory directly to train.py"
            )
        report = convert_mp20_csv(
            source,
            args.output_dir / f"{split}.jsonl",
            max_sites=args.max_sites,
            num_elements=args.num_elements,
        )
        reports.append(asdict(report))
        print(json.dumps(asdict(report), sort_keys=True))
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
