"""Command construction for an external LeMat-GenBench checkout."""

from __future__ import annotations

from pathlib import Path


def build_lemat_command(
    *,
    cifs: Path,
    config: str,
    name: str,
    uv: str = "uv",
    families: list[str] | None = None,
    batch_size: int | None = None,
) -> list[str]:
    command = [
        uv,
        "run",
        "scripts/run_benchmarks.py",
        "--cifs",
        str(Path(cifs).resolve()),
        "--config",
        config,
        "--name",
        name,
    ]
    if families:
        command.extend(["--families", *families])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    return command
