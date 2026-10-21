"""Version guard for the count-conditioned score architecture."""

from __future__ import annotations

CHECKPOINT_FORMAT_VERSION = 2
ARCHITECTURE_ID = "fixed-count-periodic-score-v4"


def validate_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION or checkpoint.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(
            "This code requires a newly trained v4 checkpoint. v1/v2/v3 checkpoints "
            "use different existence, coordinate heads and noise schedules; do not "
            "load them with strict=False. Retrain with the v4 conf/base.yaml."
        )
    for key in ("model_config", "diffusion_config", "normalization", "count_prior", "model_state"):
        if key not in checkpoint:
            raise ValueError(f"v4 checkpoint is missing {key!r}")
