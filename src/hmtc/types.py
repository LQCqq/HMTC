"""Core tensor containers and shape validation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class StructureState:
    """Structure-only view exposed to an independent energy evaluator.

    It intentionally contains no generated energy or composition factor, which
    prevents accidental target leakage through the evaluator interface.
    """

    lattice: Tensor
    elements: Tensor
    frac_coords: Tensor
    existence: Tensor
    site_mask: Tensor | None = None


@dataclass
class CrystalState:
    """Joint state diffused by HMTF.

    Element indices use ``0 .. num_elements - 1`` for real species and
    ``mask_token == num_elements`` only in noisy states.  ``site_mask`` marks
    allocated computational slots; ``existence`` is the generated physical
    occupancy and is deliberately a separate factor.
    """

    energy: Tensor
    lattice: Tensor
    composition: Tensor
    elements: Tensor
    frac_coords: Tensor
    existence: Tensor
    site_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.elements.shape[0])

    @property
    def max_sites(self) -> int:
        return int(self.elements.shape[1])

    def resolved_site_mask(self) -> Tensor:
        if self.site_mask is None:
            return torch.ones_like(self.existence, dtype=torch.bool)
        return self.site_mask.bool()

    def validate(self, num_elements: int, allow_mask_token: bool = False) -> None:
        batch, sites = self.elements.shape
        expected = {
            "energy": (batch, 1),
            "lattice": (batch, 6),
            "composition": (batch, num_elements),
            "frac_coords": (batch, sites, 3),
            "existence": (batch, sites),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}"
                )
        if self.site_mask is not None and tuple(self.site_mask.shape) != (batch, sites):
            raise ValueError("site_mask must have shape [batch, sites]")
        upper = num_elements if allow_mask_token else num_elements - 1
        if bool((self.elements < 0).any()) or bool((self.elements > upper).any()):
            raise ValueError(f"element indices must lie in [0, {upper}]")

    def to(self, *args: object, **kwargs: object) -> CrystalState:
        values = {}
        for name in (
            "energy",
            "lattice",
            "composition",
            "elements",
            "frac_coords",
            "existence",
            "site_mask",
        ):
            value = getattr(self, name)
            values[name] = None if value is None else value.to(*args, **kwargs)
        return replace(self, **values)

    def detach(self) -> CrystalState:
        return self.to_detached()

    def to_detached(self) -> CrystalState:
        values = {}
        for name in (
            "energy",
            "lattice",
            "composition",
            "elements",
            "frac_coords",
            "existence",
            "site_mask",
        ):
            value = getattr(self, name)
            values[name] = None if value is None else value.detach()
        return replace(self, **values)

    def clone(self) -> CrystalState:
        values = {}
        for name in (
            "energy",
            "lattice",
            "composition",
            "elements",
            "frac_coords",
            "existence",
            "site_mask",
        ):
            value = getattr(self, name)
            values[name] = None if value is None else value.clone()
        return replace(self, **values)


@dataclass(frozen=True)
class NoiseRecord:
    """Noise retained for diagnostics and score-training extensions."""

    energy: Tensor
    lattice: Tensor
    composition: Tensor
    coordinates: Tensor
    element_mask: Tensor
    existence_mask: Tensor
