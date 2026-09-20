import torch

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
