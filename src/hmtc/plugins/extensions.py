"""Schemas for future non-core trees.

These interfaces deliberately do not claim to implement defect thermodynamics
or synthesis planning.  They define the typed entities and bridges that future
plugins must expose to the shared workspace.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtensionInterface:
    name: str
    leaf_entity: str
    bridge_entities: tuple[str, ...]
    predicted_factors: tuple[str, ...]


DEFECT_INTERFACE = ExtensionInterface(
    name="defect",
    leaf_entity="defect_candidate",
    bridge_entities=("material", "site", "lattice", "energy", "supercell"),
    predicted_factors=("operation", "charge_state", "defect_energy", "local_displacement"),
)

RECIPE_INTERFACE = ExtensionInterface(
    name="recipe",
    leaf_entity="process_step",
    bridge_entities=("material", "composition", "phase", "energy", "precursor"),
    predicted_factors=("precursor", "operation", "temperature", "pressure", "duration"),
)
