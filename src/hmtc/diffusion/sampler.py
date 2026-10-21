"""Fixed-count sampling with absorbing masks and a wrapped-VE predictor."""

from __future__ import annotations

import torch
from torch import Tensor

from hmtc.config import DiffusionConfig, HMTFConfig
from hmtc.diffusion.schedules import coordinate_sigma, ddim_step, wrapped_ve_step
from hmtc.models.hmtf import HMTFModel
from hmtc.types import CrystalState


class HMTFSampler:
    def __init__(
        self,
        model: HMTFModel,
        model_config: HMTFConfig,
        diffusion_config: DiffusionConfig,
        count_prior: Tensor | list[float] | None = None,
    ) -> None:
        if model.diffusion_config != diffusion_config:
            raise ValueError("model and sampler must use the same diffusion configuration")
        self.model = model
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.count_prior = None if count_prior is None else torch.as_tensor(count_prior, dtype=torch.float32)
        if self.count_prior is not None:
            if self.count_prior.shape != (model_config.max_sites,):
                raise ValueError("count_prior must contain probabilities for N=1..max_sites")
            if not torch.isfinite(self.count_prior).all() or (self.count_prior < 0).any() or self.count_prior.sum() <= 0:
                raise ValueError("count_prior must contain finite nonnegative probabilities with positive mass")
            self.count_prior = self.count_prior / self.count_prior.sum()
        self.last_diagnostics: dict[str, object] = {}

    def _counts(self, batch_size: int, device: torch.device | str, num_atoms: int | Tensor | None) -> Tensor:
        if num_atoms is None:
            if self.count_prior is None:
                raise ValueError("supply the checkpoint's training count_prior or an explicit num_atoms condition")
            counts = torch.multinomial(self.count_prior.to(device), batch_size, replacement=True) + 1
        else:
            raw = torch.as_tensor(num_atoms, device=device)
            if raw.is_floating_point() and bool((raw != raw.round()).any()):
                raise ValueError("num_atoms must be integer-valued")
            counts = raw.long()
            if counts.ndim == 0:
                counts = counts.expand(batch_size)
        if counts.shape != (batch_size,) or bool(((counts < 1) | (counts > self.model_config.max_sites)).any()):
            raise ValueError("num_atoms must have one value per sample in [1, max_sites]")
        return counts

    def initial_state(self, batch_size: int, device: torch.device | str, num_atoms: int | Tensor | None = None) -> CrystalState:
        cfg = self.model_config
        counts = self._counts(batch_size, device, num_atoms)
        active = torch.arange(cfg.max_sites, device=device)[None] < counts[:, None]
        elements = torch.full((batch_size, cfg.max_sites), cfg.num_elements, device=device, dtype=torch.long)
        frac_coords = torch.rand(batch_size, cfg.max_sites, 3, device=device)
        return CrystalState(
            energy=torch.randn(batch_size, 1, device=device),
            lattice=self.model.normalization.denormalize_lattice(torch.randn(batch_size, 6, device=device)),
            composition=torch.randn(batch_size, cfg.num_elements, device=device),
            elements=torch.where(active, elements, torch.zeros_like(elements)),
            frac_coords=torch.where(active[..., None], frac_coords, torch.zeros_like(frac_coords)),
            existence=active.float(),
            site_mask=active,
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
        num_atoms: int | Tensor | None = None,
    ) -> CrystalState:
        steps = self.diffusion_config.timesteps if steps is None else steps
        if batch_size < 1 or steps < 2 or temperature <= 0:
            raise ValueError("batch_size > 0, steps >= 2 and temperature > 0 are required")
        if energy_clamp is not None:
            raise ValueError("legacy energy_clamp is not trained conditional generation; use unconditional v4 sampling")
        self.model.eval()
        state = self.initial_state(batch_size, device, num_atoms)
        active = state.physical_site_mask()
        # Match the lowest training noise level, then finish with a clean mean
        # update rather than an untrained t=0 model call.
        times = torch.cat((torch.linspace(1.0, 1.0 / self.diffusion_config.timesteps, steps, device=device),
                           torch.zeros(1, device=device)))
        image_hits = torch.zeros((), device=device)
        stats = self.model.normalization
        for index in range(steps):
            time = times[index].expand(batch_size)
            next_time = times[index + 1].expand(batch_size)
            output = self.model(state, time, active_trees=active_trees)
            image_hits += output.geometry_image_limit_hit.sum()
            energy = ddim_step(state.energy, output.energy, time, next_time,
                               self.diffusion_config.energy_schedule_power)
            lattice = stats.denormalize_lattice(ddim_step(
                stats.normalize_lattice(state.lattice), stats.normalize_lattice(output.lattice),
                time, next_time, self.diffusion_config.lattice_schedule_power))
            composition = ddim_step(state.composition, output.composition, time, next_time,
                                    self.diffusion_config.composition_schedule_power)
            next_sigma = coordinate_sigma(next_time, self.diffusion_config.coordinate_sigma_min,
                                          self.diffusion_config.coordinate_sigma_max)
            frac_coords = wrapped_ve_step(state.frac_coords, output.coordinate_score,
                                         output.coordinate_sigma, next_sigma)
            unmask_probability = ((time - next_time) / time.clamp_min(1.0e-8))[:, None]
            should_sample = active & (state.elements == self.model_config.num_elements)
            should_sample &= torch.rand_like(state.existence) < unmask_probability
            probability = torch.softmax(output.element_logits / temperature, dim=-1)
            sampled = torch.multinomial(probability.reshape(-1, probability.shape[-1]), 1).reshape_as(state.elements)
            elements = torch.where(should_sample, sampled, state.elements)
            state = CrystalState(
                energy=energy,
                lattice=lattice,
                composition=composition,
                elements=elements,
                frac_coords=torch.where(active[..., None], frac_coords, torch.zeros_like(frac_coords)),
                existence=active.to(state.existence.dtype),
                site_mask=active,
            )
        if bool((state.elements[active] == self.model_config.num_elements).any()):
            raise RuntimeError("absorbing-mask sampling left active atoms masked")
        self.last_diagnostics = {
            "num_atoms": state.num_atoms.cpu().tolist(),
            "geometry_image_limit_fraction": float((image_hits / (batch_size * steps)).cpu()),
        }
        return state
