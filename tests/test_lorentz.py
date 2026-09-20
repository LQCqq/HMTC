import torch

from hmtc.manifolds.lorentz import expmap0, logmap0, minkowski_dot, pairwise_lorentz_distance


def test_expmap_points_lie_on_unit_hyperboloid() -> None:
    tangent = torch.randn(3, 5, 8) * 0.2
    point = expmap0(tangent)
    norm = minkowski_dot(point, point)
    assert torch.allclose(norm, torch.full_like(norm, -1.0), atol=1.0e-5)


def test_origin_maps_are_locally_inverse() -> None:
    tangent = torch.randn(4, 7) * 0.2
    recovered = logmap0(expmap0(tangent))
    assert torch.allclose(recovered, tangent, atol=1.0e-5)


def test_pairwise_distance_shape_and_diagonal() -> None:
    points = expmap0(torch.randn(2, 4, 6) * 0.1)
    distances = pairwise_lorentz_distance(points, points)
    assert distances.shape == (2, 4, 4)
    assert torch.all(distances.diagonal(dim1=-2, dim2=-1) < 1.0e-3)
