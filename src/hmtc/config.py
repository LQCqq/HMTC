"""Typed configuration for the initial HMTF implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TreeSpec:
    """Description of one registered hierarchy.

    The model accepts a variable-length list of these specifications.  The two
    core trees are merely a default configuration, not an architectural limit.
    """

    name: str
    kind: str
    groups: tuple[int, ...]
    leaf_entity: str = "site"
    output_factor: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tree name must be non-empty")
        if not self.groups or any(group <= 0 for group in self.groups):
            raise ValueError("groups must contain positive sizes")


@dataclass(frozen=True)
class HMTFConfig:
    num_elements: int = 100
    max_sites: int = 20
    hidden_dim: int = 64
    num_global_tokens: int = 4
    num_blocks: int = 4
    num_rbf: int = 16
    rbf_cutoff: float = 8.0
    geometry_layers: int = 3
    geometry_refinement_layers: int = 1
    periodic_image_limit: int = 3
    tree_specs: tuple[TreeSpec, ...] = field(
        default_factory=lambda: (
            TreeSpec("chemical", "chemical", (8, 4), output_factor="element"),
            TreeSpec("geometry", "geometry", (8, 4), output_factor="coordinate"),
        )
    )

    def __post_init__(self) -> None:
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if self.max_sites <= 0 or self.num_elements <= 0:
            raise ValueError("max_sites and num_elements must be positive")
        if self.geometry_layers < 1 or self.geometry_refinement_layers < 1:
            raise ValueError("geometry and refinement paths must each have at least one layer")
        if self.periodic_image_limit < 1 or self.rbf_cutoff <= 0:
            raise ValueError("periodic_image_limit and rbf_cutoff must be positive")
        names = [spec.name for spec in self.tree_specs]
        if len(names) != len(set(names)):
            raise ValueError("tree names must be unique")


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    coordinate_sigma_min: float = 0.01
    coordinate_sigma_max: float = 1.0
    categorical_mask_probability_max: float = 1.0
    energy_schedule_power: float = 1.60
    lattice_schedule_power: float = 1.15
    composition_schedule_power: float = 1.30

    def __post_init__(self) -> None:
        if self.timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0 < self.coordinate_sigma_min < self.coordinate_sigma_max:
            raise ValueError("require 0 < coordinate_sigma_min < coordinate_sigma_max")
        if self.coordinate_sigma_max < 1.0:
            raise ValueError("coordinate_sigma_max >= 1 makes the terminal torus density uniform to float32 tolerance")
        if self.categorical_mask_probability_max != 1.0:
            raise ValueError("absorbing-mask diffusion requires a fully masked terminal state")


@dataclass(frozen=True)
class LossConfig:
    energy: float = 1.0
    lattice: float = 1.0
    distance: float = 1.0
    composition: float = 0.5
    element: float = 1.0
    element_count: float = 0.5
    coordinate: float = 2.0
    existence: float = 0.0
    energy_consistency: float = 0.25
    composition_consistency: float = 0.25
    routing_balance: float = 0.05
    routing_entropy: float = 0.005
    tree_redundancy: float = 0.10
    min_distance_offset: float = 0.7
    min_distance_scaling_factor: float = 0.5
    tree_redundancy_margin: float = 0.7
    geometry_loss_max_time: float = 0.5

    def __post_init__(self) -> None:
        if self.min_distance_offset < 0.0:
            raise ValueError("min_distance_offset must be non-negative")
        if self.min_distance_scaling_factor <= 0.0:
            raise ValueError("min_distance_scaling_factor must be positive")
        if not 0.0 <= self.tree_redundancy_margin < 1.0:
            raise ValueError("tree_redundancy_margin must lie in [0, 1)")
        if self.existence != 0.0:
            raise ValueError("existence is fixed by count conditioning; its loss weight must be zero")
        if not 0.0 < self.geometry_loss_max_time <= 1.0:
            raise ValueError("geometry_loss_max_time must lie in (0, 1]")


def _tree_spec(raw: dict[str, Any]) -> TreeSpec:
    item = dict(raw)
    item["groups"] = tuple(item["groups"])
    return TreeSpec(**item)


def load_config(path: str | Path) -> tuple[HMTFConfig, DiffusionConfig, LossConfig, dict[str, Any]]:
    """Load the checked-in YAML while retaining sampling options."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    model_raw = dict(raw["model"])
    model_raw["tree_specs"] = tuple(_tree_spec(item) for item in model_raw["tree_specs"])
    return (
        HMTFConfig(**model_raw),
        DiffusionConfig(**raw.get("diffusion", {})),
        LossConfig(**raw.get("loss", {})),
        dict(raw.get("sampling", {})),
    )
