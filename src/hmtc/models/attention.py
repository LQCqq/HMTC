"""Hyperbolic attention used by the global workspace and typed bridges."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from hmtc.manifolds.lorentz import expmap0, logmap0, pairwise_lorentz_distance


class HyperbolicAttention(nn.Module):
    """Attention scores are negative Lorentz distances; values aggregate in ``T_oH``."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(math.sqrt(hidden_dim))))

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        query_tangent = logmap0(query)
        context_tangent = logmap0(context)
        query_point = expmap0(self.query(query_tangent))
        key_point = expmap0(self.key(context_tangent))
        score = -pairwise_lorentz_distance(query_point, key_point).square()
        score = score / self.log_temperature.exp().clamp_min(1.0e-4)
        if context_mask is not None:
            score = score.masked_fill(
                ~context_mask[:, None, :].bool(), torch.finfo(score.dtype).min
            )
        attention = self.dropout(torch.softmax(score, dim=-1))
        message = torch.matmul(attention, self.value(context_tangent))
        updated = self.norm(query_tangent + self.output(message))
        return expmap0(updated)
