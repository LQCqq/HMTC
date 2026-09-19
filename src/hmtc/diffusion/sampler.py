"""CFG-free reverse generation for the joint HMTF state."""

from __future__ import annotations

import torch
from torch import Tensor

from hmtc.config import DiffusionConfig, HMTFConfig
from hmtc.diffusion.schedules import ddim_step
from hmtc.models.hmtf import HMTFModel
from hmtc.types import CrystalState


def _periodic_delta(target: Tensor, current: Tensor) -> Tensor:
    return torch.remainder(target - current + 0.5, 1.0) - 0.5


class HMTFSampler:
    """Single-network sampler; there is no conditional/unconditional CFG pair."""

    def __init__(
        self,
        model: HMTFModel,
        model_config: HMTFConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.diffusion_config = diffusion_config

    def initial_state(self, batch_size: int, device: torch.device | str) -> CrystalState:
        cfg = self.model_config
        return CrystalState(
            energy=torch.randn(batch_size, 1, device=device),
            lattice=torch.randn(batch_size, 6, device=device),
            composition=torch.randn(batch_size, cfg.num_elements, device=device),
            elements=torch.full(
                (batch_size, cfg.max_sites), cfg.num_elements, device=device, dtype=torch.long
            ),
            frac_coords=torch.rand(batch_size, cfg.max_sites, 3, device=device),
            existence=torch.bernoulli(torch.full((batch_size, cfg.max_sites), 0.5, device=device)),
            site_mask=torch.ones(batch_size, cfg.max_sites, device=device, dtype=torch.bool),
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        steps: int | None = None,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        energy_clamp: float | None = None,
        active_trees: set[str] | None = None,
    ) -> CrystalState:
        steps = steps or self.diffusion_config.timesteps
        state = self.initial_state(batch_size, device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for index in range(steps):
            time = times[index].expand(batch_size)
            next_time = times[index + 1].expand(batch_size)
            output = self.model(state, time, active_trees=active_trees)
            predicted = output.as_state(site_mask=state.site_mask)
            energy = ddim_step(
                state.energy,
                predicted.energy,
                time,
                next_time,
                self.diffusion_config.energy_schedule_power,
            )
            if energy_clamp is not None:
                energy = torch.full_like(energy, energy_clamp)
            lattice = ddim_step(
                state.lattice,
                predicted.lattice,
                time,
                next_time,
                self.diffusion_config.lattice_schedule_power,
            )
            composition = ddim_step(
                state.composition,
                predicted.composition,
                time,
                next_time,
                self.diffusion_config.composition_schedule_power,
            )
            fraction = ((time - next_time) / time.clamp_min(1.0e-6))[:, None, None]
            frac_coords = torch.remainder(
                state.frac_coords
                + fraction * _periodic_delta(predicted.frac_coords, state.frac_coords),
                1.0,
            )
            existence_fraction = fraction.squeeze(-1)
            existence = state.existence + existence_fraction * (
                predicted.existence - state.existence
            )
            unmask_probability = ((time - next_time) / time.clamp_min(1.0e-6))[:, None]
            should_sample = torch.rand_like(state.existence) < unmask_probability
            probabilities = torch.softmax(output.element_logits / max(temperature, 1.0e-4), dim=-1)
            sampled = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1
            ).reshape(batch_size, self.model_config.max_sites)
            elements = torch.where(should_sample, sampled, state.elements)
            state = CrystalState(
                energy=energy,
                lattice=lattice,
                composition=composition,
                elements=elements,
                frac_coords=frac_coords,
                existence=existence,
                site_mask=state.site_mask,
            )

        final_time = torch.zeros(batch_size, device=device)
        final = self.model(state, final_time, active_trees=active_trees).as_state(state.site_mask)
        final.existence = (final.existence >= 0.5).to(final.existence.dtype)
        return final
