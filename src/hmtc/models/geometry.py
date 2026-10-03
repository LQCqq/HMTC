"""Periodic geometry utilities and physical site messages."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hmtc.manifolds.lorentz import expmap0, logmap0

IMAGE_OFFSETS = torch.cartesian_prod(
    torch.tensor([-1.0, 0.0, 1.0]),
    torch.tensor([-1.0, 0.0, 1.0]),
    torch.tensor([-1.0, 0.0, 1.0]),
)


def lattice_params_to_matrix(params: Tensor) -> Tensor:
    """Build a positive-volume lower-triangular lattice from six parameters.

    Parameter order is ``log(a_x), b_x, log(b_y), c_x, c_y, log(c_z)``.
    Rows are lattice vectors and Cartesian coordinates are ``fractional @ L``.
    """

    matrix = params.new_zeros(*params.shape[:-1], 3, 3)
    matrix[..., 0, 0] = params[..., 0].clamp(-8.0, 8.0).exp()
    matrix[..., 1, 0] = params[..., 1]
    matrix[..., 1, 1] = params[..., 2].clamp(-8.0, 8.0).exp()
    matrix[..., 2, 0] = params[..., 3]
    matrix[..., 2, 1] = params[..., 4]
    matrix[..., 2, 2] = params[..., 5].clamp(-8.0, 8.0).exp()
    return matrix


def lattice_matrix_to_params(matrix: Tensor) -> Tensor:
    """Inverse for lower-triangular standardized cells."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("lattice matrix must end in [3, 3]")
    return torch.stack(
        (
            matrix[..., 0, 0].clamp_min(1.0e-8).log(),
            matrix[..., 1, 0],
            matrix[..., 1, 1].clamp_min(1.0e-8).log(),
            matrix[..., 2, 0],
            matrix[..., 2, 1],
            matrix[..., 2, 2].clamp_min(1.0e-8).log(),
        ),
        dim=-1,
    )


def minimum_image(delta: Tensor) -> Tensor:
    """Wrap fractional displacements to ``[-0.5, 0.5)``."""

    return delta - torch.round(delta)


def pairwise_periodic_geometry(
    frac_coords: Tensor, lattice_params: Tensor
) -> tuple[Tensor, Tensor]:
    """Return shortest displacements among the neighboring 27 periodic images.

    Component-wise wrapping is insufficient for skewed triclinic cells.  The
    27-image search is exact for standardized reduced cells and remains cheap
    for the small cells targeted by the initial model.
    """

    delta_fractional = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]
    lattice = lattice_params_to_matrix(lattice_params)
    offsets = IMAGE_OFFSETS.to(device=frac_coords.device, dtype=frac_coords.dtype)
    candidates = delta_fractional.unsqueeze(-2) - offsets
    cartesian_candidates = torch.einsum("bijnc,bcd->bijnd", candidates, lattice)
    squared_distances = cartesian_candidates.square().sum(dim=-1)
    best = squared_distances.argmin(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 3)
    displacement = cartesian_candidates.gather(-2, gather_index).squeeze(-2)
    best_squared_distances = squared_distances.gather(
        -1, best.unsqueeze(-1)
    ).squeeze(-1)
    # Self-pairs have exactly zero distance. A direct sqrt(0) has an infinite
    # derivative and can produce 0 * inf = NaN even when those pairs are masked
    # later. Clamp before sqrt, then restore exact zeros with a zero-gradient
    # branch so all downstream periodic-distance losses remain finite.
    distances = best_squared_distances.clamp_min(1.0e-12).sqrt()
    distances = torch.where(
        best_squared_distances > 0.0,
        distances,
        torch.zeros_like(distances),
    )
    return displacement, distances


class PeriodicMessageLayer(nn.Module):
    """All-pairs periodic messages for small variable-size crystal cells."""

    def __init__(self, hidden_dim: int, num_rbf: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        width = cutoff / max(num_rbf - 1, 1)
        self.gamma = 1.0 / max(width * width, 1.0e-6)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf + 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        sites: Tensor,
        frac_coords: Tensor,
        lattice_params: Tensor,
        existence: Tensor,
        site_mask: Tensor,
    ) -> Tensor:
        tangent = logmap0(sites)
        batch, num_sites, hidden = tangent.shape
        displacement, distance = pairwise_periodic_geometry(frac_coords, lattice_params)
        rbf = torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers).square())
        radial_cutoff = 0.5 * (torch.cos(torch.pi * distance / self.cutoff) + 1.0)
        radial_cutoff = radial_cutoff * (distance < self.cutoff)
        hi = tangent[:, :, None, :].expand(batch, num_sites, num_sites, hidden)
        hj = tangent[:, None, :, :].expand(batch, num_sites, num_sites, hidden)
        edge = self.edge_mlp(torch.cat((hi, hj, rbf, displacement), dim=-1))
        diagonal = torch.eye(num_sites, device=sites.device, dtype=torch.bool)[None]
        pair_mask = site_mask[:, :, None] & site_mask[:, None, :] & ~diagonal
        sender_weight = existence[:, None, :].clamp(0.0, 1.0)
        weight = pair_mask.to(edge.dtype) * sender_weight * radial_cutoff
        message = (edge * weight.unsqueeze(-1)).sum(dim=2)
        normalizer = weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        message = message / normalizer
        updated = tangent + self.update(torch.cat((tangent, message), dim=-1))
        return expmap0(updated)
