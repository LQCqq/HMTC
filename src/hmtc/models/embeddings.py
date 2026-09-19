"""Scalar, time, and periodic coordinate embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, value: Tensor) -> Tensor:
        value = value.float().reshape(*value.shape, 1)
        half = self.dim // 2
        if half == 0:
            return value
        exponent = (
            -math.log(self.max_period)
            * torch.arange(half, device=value.device, dtype=value.dtype)
            / max(half - 1, 1)
        )
        angles = value * torch.exp(exponent)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if self.dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[..., :1])), dim=-1)
        return embedding


class FourierCoordinateEmbedding(nn.Module):
    """Embedding invariant to integer translations of fractional coordinates."""

    def __init__(self, num_bands: int = 4) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_bands).float())

    @property
    def output_dim(self) -> int:
        return int(self.frequencies.numel()) * 3 * 2

    def forward(self, frac_coords: Tensor) -> Tensor:
        angles = 2.0 * math.pi * frac_coords.unsqueeze(-1) * self.frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)


class FactorEncoder(nn.Module):
    """Encode one continuous global factor into a tangent vector."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)
