"""Numerically conservative Lorentz-model primitives.

Physical positions never live on this manifold.  HMTF uses Lorentz space only
for semantic, factor, hierarchy, and workspace representations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

EPS = 1.0e-7
MAX_TANGENT_NORM = 15.0


def _safe_norm(value: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    return value.norm(dim=dim, keepdim=keepdim).clamp_min(EPS)


def minkowski_dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    """Lorentzian inner product with signature ``(-,+,...,+)``."""

    result = -(x[..., :1] * y[..., :1]).sum(dim=-1, keepdim=True)
    result = result + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return result if keepdim else result.squeeze(-1)


def project_lorentz(x: Tensor) -> Tensor:
    """Project ambient vectors onto the upper unit hyperboloid."""

    spatial = x[..., 1:]
    time = torch.sqrt(1.0 + spatial.square().sum(dim=-1, keepdim=True))
    return torch.cat((time, spatial), dim=-1)


def expmap0(tangent: Tensor) -> Tensor:
    """Map an origin-tangent spatial vector to the unit hyperboloid."""

    raw_norm = _safe_norm(tangent)
    norm = raw_norm.clamp_max(MAX_TANGENT_NORM)
    direction = tangent / raw_norm
    spatial = torch.sinh(norm) * direction
    time = torch.cosh(norm)
    return torch.cat((time, spatial), dim=-1)


def logmap0(point: Tensor) -> Tensor:
    """Map a hyperboloid point to the spatial coordinates of ``T_o H``."""

    point = project_lorentz(point)
    distance = torch.acosh(point[..., :1].clamp_min(1.0 + EPS))
    spatial = point[..., 1:]
    return distance * spatial / _safe_norm(spatial)


def lorentz_distance(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    argument = (-minkowski_dot(project_lorentz(x), project_lorentz(y))).clamp_min(1.0 + EPS)
    distance = torch.acosh(argument)
    return distance.unsqueeze(-1) if keepdim else distance


def pairwise_lorentz_distance(x: Tensor, y: Tensor) -> Tensor:
    """Return distances between ``[..., N, D+1]`` and ``[..., M, D+1]``."""

    x = project_lorentz(x)
    y = project_lorentz(y)
    product = -(x[..., :, :1] * y[..., :, :1].transpose(-1, -2))
    product = product + torch.matmul(x[..., :, 1:], y[..., :, 1:].transpose(-1, -2))
    return torch.acosh((-product).clamp_min(1.0 + EPS))


def tangent_mean(points: Tensor, weights: Tensor | None = None, dim: int = -2) -> Tensor:
    """Stable approximation to a hyperbolic barycenter via the origin tangent."""

    tangent = logmap0(points)
    if weights is None:
        mean = tangent.mean(dim=dim)
    else:
        weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(EPS)
        mean = (tangent * weights.unsqueeze(-1)).sum(dim=dim)
    return expmap0(mean)


class HyperbolicLinear(nn.Module):
    """A tangent-space affine transform with a Lorentz-valued output."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, point: Tensor) -> Tensor:
        return expmap0(self.linear(logmap0(point)))


class HyperbolicResidualMLP(nn.Module):
    """Residual MLP whose public input and output remain on the hyperboloid."""

    def __init__(self, hidden_dim: int, expansion: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(self, point: Tensor, message: Tensor | None = None) -> Tensor:
        tangent = logmap0(point)
        if message is not None:
            tangent = tangent + message
        return expmap0(tangent + self.net(tangent))
