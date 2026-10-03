import torch

from hmtc.routing import batch_routing_occupancy, off_diagonal_coassignment


def test_batch_occupancy_can_balance_individually_collapsed_samples() -> None:
    assignment = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    child_weight = torch.ones(2, 1)
    occupancy = batch_routing_occupancy(assignment, child_weight)
    assert torch.allclose(occupancy, torch.tensor([0.5, 0.5]))


def test_off_diagonal_coassignment_excludes_trivial_self_pairs() -> None:
    separate = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    together = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    child_weight = torch.ones(1, 2)
    separate_pairs = off_diagonal_coassignment(separate, child_weight)
    together_pairs = off_diagonal_coassignment(together, child_weight)
    assert torch.equal(separate_pairs, torch.zeros(1, 2))
    assert torch.equal(together_pairs, torch.ones(1, 2))
