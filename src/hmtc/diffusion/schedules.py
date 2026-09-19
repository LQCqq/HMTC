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
