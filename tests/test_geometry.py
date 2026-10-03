import torch

from hmtc.losses import minimum_distance_penalty
from hmtc.models.geometry import (
    lattice_matrix_to_params,
    lattice_params_to_matrix,
    minimum_image,
    pairwise_periodic_geometry,
)


def test_lattice_parameterization_has_positive_volume() -> None:
    params = torch.randn(5, 6)
    lattice = lattice_params_to_matrix(params)
    assert torch.all(torch.linalg.det(lattice) > 0)
    recovered = lattice_matrix_to_params(lattice)
    assert torch.allclose(recovered, params, atol=1.0e-5)


def test_minimum_image_and_pairwise_symmetry() -> None:
    delta = torch.tensor([0.9, -0.6, 0.2])
    wrapped = minimum_image(delta)
    assert torch.all(wrapped >= -0.5)
    assert torch.all(wrapped < 0.5)
    coords = torch.rand(2, 5, 3)
    params = torch.zeros(2, 6)
    _, distances = pairwise_periodic_geometry(coords, params)
    assert torch.allclose(distances, distances.transpose(1, 2), atol=1.0e-6)


def test_minimum_distance_penalty_and_joint_gradients() -> None:
    logits = torch.zeros(1, 2, 100, requires_grad=True)
    close_coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]], requires_grad=True
    )
    lattice = torch.tensor(
        [[1.3862944, 0.0, 1.3862944, 0.0, 0.0, 1.3862944]],
        requires_grad=True,
    )
    active = torch.tensor([[True, True]])
    penalty = minimum_distance_penalty(logits, close_coords, lattice, active)
    assert penalty > 0
    penalty.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert close_coords.grad is not None and close_coords.grad.abs().sum() > 0
    assert lattice.grad is not None and lattice.grad.abs().sum() > 0


def test_minimum_distance_penalty_is_zero_for_separated_pair() -> None:
    logits = torch.full((1, 2, 100), -20.0)
    logits[..., 0] = 20.0
    separated_coords = torch.tensor([[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]])
    lattice = torch.tensor([[1.3862944, 0.0, 1.3862944, 0.0, 0.0, 1.3862944]])
    active = torch.tensor([[True, True]])
    penalty = minimum_distance_penalty(logits, separated_coords, lattice, active)
    assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1.0e-7)
