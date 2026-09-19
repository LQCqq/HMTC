"""Joint corruption of energy, lattice, composition, and atomic site factors."""

from __future__ import annotations

import torch
from torch import Tensor

from hmtc.config import DiffusionConfig
from hmtc.diffusion.schedules import gaussian_q_sample
from hmtc.types import CrystalState, NoiseRecord


class JointDiffusion:
    def __init__(self, config: DiffusionConfig, num_elements: int) -> None:
        self.config = config
        self.num_elements = num_elements
        self.mask_token = num_elements

    def q_sample(self, clean: CrystalState, time: Tensor) -> tuple[CrystalState, NoiseRecord]:
        if time.shape != (clean.batch_size,):
            raise ValueError("time must have shape [batch]")
        energy, energy_noise = gaussian_q_sample(
            clean.energy, time, self.config.energy_schedule_power
        )
        lattice, lattice_noise = gaussian_q_sample(
            clean.lattice, time, self.config.lattice_schedule_power
        )
        composition, composition_noise = gaussian_q_sample(
            clean.composition, time, self.config.composition_schedule_power
        )
        coordinate_noise = torch.randn_like(clean.frac_coords)
        coordinate_sigma = self.config.coordinate_sigma_max * torch.sin(time * torch.pi / 2.0)
        frac_coords = torch.remainder(
            clean.frac_coords + coordinate_sigma[:, None, None] * coordinate_noise, 1.0
        )
        categorical_probability = (time * self.config.categorical_mask_probability_max).clamp(
            0.0, 1.0
        )
        element_mask = torch.rand_like(clean.existence) < categorical_probability[:, None]
        element_mask = element_mask & clean.resolved_site_mask()
        elements = torch.where(element_mask, self.mask_token, clean.elements)
        existence_mask = torch.rand_like(clean.existence) < time[:, None]
        random_existence = torch.bernoulli(torch.full_like(clean.existence, 0.5))
        existence = torch.where(existence_mask, random_existence, clean.existence)
        noisy = CrystalState(
            energy=energy,
            lattice=lattice,
            composition=composition,
            elements=elements,
            frac_coords=frac_coords,
            existence=existence,
            site_mask=clean.site_mask,
        )
        record = NoiseRecord(
            energy=energy_noise,
            lattice=lattice_noise,
            composition=composition_noise,
            coordinates=coordinate_noise,
            element_mask=element_mask,
            existence_mask=existence_mask,
        )
        return noisy, record
