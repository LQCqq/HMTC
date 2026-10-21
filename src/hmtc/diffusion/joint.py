"""Joint corruption of energy, lattice, composition, and atomic site factors."""

from __future__ import annotations

import math
import torch
from torch import Tensor

from hmtc.config import DiffusionConfig
from hmtc.data.schema import NormalizationStats
from hmtc.diffusion.schedules import coordinate_sigma, gaussian_q_sample, wrapped_normal_score
from hmtc.types import CrystalState, NoiseRecord


class JointDiffusion:
    def __init__(self, config: DiffusionConfig, num_elements: int, stats: NormalizationStats | None = None) -> None:
        self.config = config
        self.num_elements = num_elements
        self.mask_token = num_elements
        self.stats = stats or NormalizationStats(0.0, 1.0)

    def q_sample(self, clean: CrystalState, time: Tensor) -> tuple[CrystalState, NoiseRecord]:
        if time.shape != (clean.batch_size,):
            raise ValueError("time must have shape [batch]")
        active = clean.physical_site_mask()
        energy, energy_noise = gaussian_q_sample(
            clean.energy, time, self.config.energy_schedule_power
        )
        lattice, lattice_noise = gaussian_q_sample(
            self.stats.normalize_lattice(clean.lattice), time, self.config.lattice_schedule_power
        )
        lattice = self.stats.denormalize_lattice(lattice)
        composition, composition_noise = gaussian_q_sample(
            clean.composition, time, self.config.composition_schedule_power
        )
        coordinate_noise = torch.randn_like(clean.frac_coords)
        sigma = coordinate_sigma(time, self.config.coordinate_sigma_min, self.config.coordinate_sigma_max)
        frac_coords = torch.remainder(
            clean.frac_coords + sigma[:, None, None] * coordinate_noise, 1.0
        )
        frac_coords = torch.where(active[..., None], frac_coords, torch.zeros_like(frac_coords))
        score = wrapped_normal_score(frac_coords, clean.frac_coords, sigma,
                                     image_radius=math.ceil(8 * self.config.coordinate_sigma_max) + 1)
        score = score * active[..., None]
        categorical_probability = time.clamp(0.0, 1.0)
        element_mask = torch.rand_like(clean.existence) < categorical_probability[:, None]
        element_mask = element_mask & active
        elements = torch.where(element_mask, self.mask_token, clean.elements)
        elements = torch.where(active, elements, torch.zeros_like(elements))
        existence_mask = torch.zeros_like(active)
        noisy = CrystalState(
            energy=energy,
            lattice=lattice,
            composition=composition,
            elements=elements,
            frac_coords=frac_coords,
            existence=active.to(clean.existence.dtype),
            site_mask=active,
        )
        record = NoiseRecord(
            energy=energy_noise,
            lattice=lattice_noise,
            composition=composition_noise,
            coordinates=coordinate_noise,
            element_mask=element_mask,
            existence_mask=existence_mask,
            coordinate_score=score,
            coordinate_sigma=sigma,
        )
        return noisy, record
