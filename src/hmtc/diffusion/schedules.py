"""Noise schedules for heterogeneous material factors."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_alpha_bar(time: Tensor, power: float = 1.0, offset: float = 0.008) -> Tensor:
    """Cosine cumulative signal with an optional coarse-to-fine exponent.

    For ``power > 1``, a factor retains more signal at intermediate forward
    times and therefore becomes reliable earlier during reverse generation.
    Every factor is still fully corrupted at ``time == 1``.
    """

    effective_time = time.clamp(0.0, 1.0).pow(power)
    angle = (effective_time + offset) / (1.0 + offset) * math.pi / 2.0
    initial = math.cos(offset / (1.0 + offset) * math.pi / 2.0) ** 2
    return (torch.cos(angle).square() / initial).clamp(0.0, 1.0)


def broadcast_batch(value: Tensor, target: Tensor) -> Tensor:
    return value.reshape(value.shape[0], *((1,) * (target.ndim - 1)))


def gaussian_q_sample(clean: Tensor, time: Tensor, power: float = 1.0) -> tuple[Tensor, Tensor]:
    noise = torch.randn_like(clean)
    alpha = broadcast_batch(cosine_alpha_bar(time, power), clean)
    noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
    return noisy, noise


def ddim_step(
    current: Tensor, predicted_clean: Tensor, time: Tensor, next_time: Tensor, power: float
) -> Tensor:
    alpha = broadcast_batch(cosine_alpha_bar(time, power), current)
    next_alpha = broadcast_batch(cosine_alpha_bar(next_time, power), current)
    denominator = (1.0 - alpha).sqrt().clamp_min(1.0e-6)
    estimated_noise = (current - alpha.sqrt() * predicted_clean) / denominator
    return next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * estimated_noise


def coordinate_sigma(time: Tensor, sigma_min: float, sigma_max: float) -> Tensor:
    """VE schedule; t=0 is clean, positive times follow a log-spaced scale."""
    sigma = sigma_min * (sigma_max / sigma_min) ** time.clamp(0.0, 1.0)
    return torch.where(time > 0, sigma, torch.zeros_like(sigma))


def wrapped_normal_score(noisy: Tensor, clean: Tensor, sigma: Tensor, image_radius: int = 9) -> Tensor:
    """Conditional score of a unit-period wrapped Normal, not unwrapped noise.

    The product density factorizes across fractional components. Sum enough
    integer images to capture eight standard deviations; use float64 here to
    avoid cancellation when the terminal density is essentially uniform.
    """
    delta = torch.remainder((noisy - clean).double() + 0.5, 1.0) - 0.5
    scale = broadcast_batch(sigma.double().clamp_min(1.0e-8), delta)
    images = torch.arange(-image_radius, image_radius + 1, device=noisy.device, dtype=torch.float64)
    displacement = delta.unsqueeze(-1) + images
    probability = torch.softmax(-0.5 * (displacement / scale.unsqueeze(-1)).square(), dim=-1)
    score = -(probability * displacement).sum(dim=-1) / scale.square()
    score = torch.where(broadcast_batch(sigma > 0, score), score, torch.zeros_like(score))
    return score.to(noisy.dtype)


def wrapped_ve_step(current: Tensor, score: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor:
    """Ancestral VE predictor for the same variance increments used in q."""
    variance = sigma.square()
    next_variance = next_sigma.square()
    increment = (variance - next_variance).clamp_min(0.0)
    mean = current + broadcast_batch(increment, current) * score
    posterior_variance = next_variance * increment / variance.clamp_min(1.0e-12)
    noise = broadcast_batch(posterior_variance.sqrt(), current) * torch.randn_like(current)
    return torch.remainder(mean + noise, 1.0)
