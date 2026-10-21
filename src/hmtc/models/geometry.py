"""Periodic geometry utilities and physical site messages."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class PeriodicGraph:
    """Directed radius-graph edges, including nonzero self periodic images."""

    batch_index: Tensor
    receiver: Tensor
    sender: Tensor
    displacement: Tensor
    distance: Tensor
    image_limit_hit: Tensor
    batch_size: int
    num_sites: int


def periodic_radius_graph(
    frac_coords: Tensor,
    lattice_params: Tensor,
    active: Tensor,
    cutoff: float,
    image_limit: int = 3,
) -> PeriodicGraph:
    """Enumerate periodic neighbors rather than retaining one nearest image.

    Reciprocal-cell heights determine the required integer-image ranges.
    Extremely small/skewed noisy cells may exceed the configured limit; that
    approximation is reported through image_limit_hit, never hidden. Centered
    pair displacements make the finite search invariant to origin shifts.
    """
    lattice = lattice_params_to_matrix(lattice_params)
    reciprocal_norm = torch.linalg.inv(lattice.detach()).norm(dim=-2)
    required = torch.ceil(cutoff * reciprocal_norm + 0.5).long()
    image_limit_hit = (required > image_limit).any(dim=-1)
    ranges = required.clamp(min=1, max=image_limit).amax(dim=0).tolist()
    offsets = torch.cartesian_prod(*[
        torch.arange(-extent, extent + 1, device=frac_coords.device, dtype=frac_coords.dtype)
        for extent in ranges
    ])
    delta = minimum_image(frac_coords[:, None, :, :] - frac_coords[:, :, None, :])
    candidates = delta.unsqueeze(-2) + offsets
    cartesian = torch.einsum("bijnc,bcd->bijnd", candidates, lattice)
    squared = cartesian.square().sum(dim=-1)
    pair_mask = active[:, :, None, None] & active[:, None, :, None]
    valid = pair_mask & (squared > 1.0e-12) & (squared < cutoff * cutoff)
    batch_index, receiver, sender, image = valid.nonzero(as_tuple=True)
    displacement = cartesian[batch_index, receiver, sender, image]
    distance = squared[batch_index, receiver, sender, image].clamp_min(1.0e-12).sqrt()
    return PeriodicGraph(
        batch_index=batch_index,
        receiver=receiver,
        sender=sender,
        displacement=displacement,
        distance=distance,
        image_limit_hit=image_limit_hit,
        batch_size=frac_coords.shape[0],
        num_sites=frac_coords.shape[1],
    )


class PeriodicMessageLayer(nn.Module):
    """Invariant scalar messages on a periodic radius graph."""

    def __init__(self, hidden_dim: int, num_rbf: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        width = cutoff / max(num_rbf - 1, 1)
        self.gamma = 1.0 / max(width * width, 1.0e-6)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim * 2),
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
        graph: PeriodicGraph | None = None,
    ) -> Tensor:
        tangent = logmap0(sites)
        batch, num_sites, hidden = tangent.shape
        if graph is None:
            graph = periodic_radius_graph(frac_coords, lattice_params,
                                          site_mask & existence.bool(), self.cutoff)
        rbf = torch.exp(-self.gamma * (graph.distance.unsqueeze(-1) - self.centers).square())
        weight = 0.5 * (torch.cos(torch.pi * graph.distance / self.cutoff) + 1.0)
        hi = tangent[graph.batch_index, graph.receiver]
        hj = tangent[graph.batch_index, graph.sender]
        edge = self.edge_mlp(torch.cat((hi, hj, rbf), dim=-1))
        index = graph.batch_index * num_sites + graph.receiver
        message = tangent.new_zeros(batch * num_sites, hidden)
        message.index_add_(0, index, edge * weight[:, None])
        normalizer = tangent.new_zeros(batch * num_sites, 1)
        normalizer.index_add_(0, index, weight[:, None])
        message = (message / normalizer.clamp_min(1.0)).reshape(batch, num_sites, hidden)
        updated = tangent + self.update(torch.cat((tangent, message), dim=-1))
        return expmap0(updated)


class EquivariantCoordinateHead(nn.Module):
    """Scalar edge coefficients times Cartesian directions yield vectors.

    This vector readout is E(3)-equivariant for invariant site features. It
    produces a Cartesian tangent vector, converted to fractional components
    using the cell inverse. No component-wise MLP or absolute positions occur.
    """

    def __init__(self, input_dim: int, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        width = cutoff / max(num_rbf - 1, 1)
        self.gamma = 1.0 / max(width * width, 1.0e-6)
        self.coefficient = nn.Sequential(
            nn.Linear(input_dim * 2 + num_rbf, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, 1),
        )

    def cartesian_vector(self, features: Tensor, graph: PeriodicGraph) -> Tensor:
        rbf = torch.exp(-self.gamma * (graph.distance[:, None] - self.centers).square())
        hi = features[graph.batch_index, graph.receiver]
        hj = features[graph.batch_index, graph.sender]
        coefficient = self.coefficient(torch.cat((hi, hj, rbf), dim=-1))
        cutoff_weight = 0.5 * (torch.cos(torch.pi * graph.distance / self.cutoff) + 1.0)
        direction = graph.displacement / graph.distance[:, None].clamp_min(1.0e-6)
        message = coefficient * cutoff_weight[:, None] * direction
        index = graph.batch_index * graph.num_sites + graph.receiver
        vector = features.new_zeros(graph.batch_size * graph.num_sites, 3)
        vector.index_add_(0, index, message)
        normalizer = features.new_zeros(graph.batch_size * graph.num_sites, 1)
        normalizer.index_add_(0, index, cutoff_weight[:, None])
        vector = vector / normalizer.clamp_min(1.0)
        return vector.reshape(graph.batch_size, graph.num_sites, 3)

    def forward(self, features: Tensor, graph: PeriodicGraph, lattice_params: Tensor) -> Tensor:
        vector = self.cartesian_vector(features, graph)
        inverse = torch.linalg.inv(lattice_params_to_matrix(lattice_params))
        return torch.einsum("bsc,bcd->bsd", vector, inverse)
