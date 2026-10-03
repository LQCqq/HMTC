import torch

from hmtc.config import HMTFConfig, TreeSpec
from hmtc.data import synthetic_batch
from hmtc.diagnostics import routing_diagnostics
from hmtc.models import HMTFModel


def test_routing_diagnostics_have_normalized_occupancy_and_fusion() -> None:
    config = HMTFConfig(
        num_elements=8,
        max_sites=5,
        hidden_dim=16,
        num_global_tokens=2,
        num_blocks=1,
        num_rbf=4,
        tree_specs=(
            TreeSpec("chemical", "chemical", (3, 2)),
            TreeSpec("geometry", "geometry", (3, 2)),
        ),
    )
    state = synthetic_batch(3, config.max_sites, config.num_elements)
    model = HMTFModel(config)
    output = model(state, torch.tensor([0.2, 0.5, 0.8]))
    active = state.resolved_site_mask() & (state.existence > 0.5)
    diagnostics = routing_diagnostics(output, active)

    for tree_name in ("chemical", "geometry"):
        level1 = diagnostics[f"routing.{tree_name}.level1.occupancy"]
        level2 = diagnostics[f"routing.{tree_name}.level2.occupancy"]
        assert level1.shape == (3,)
        assert level2.shape == (2,)
        assert torch.allclose(level1.sum(), torch.tensor(1.0), atol=1.0e-5)
        assert torch.allclose(level2.sum(), torch.tensor(1.0), atol=1.0e-5)
        assert torch.isfinite(diagnostics[f"routing.{tree_name}.level1.entropy"])
        assert torch.isfinite(diagnostics[f"routing.{tree_name}.level2.entropy"])

    similarity1 = diagnostics["routing_similarity.chemical__geometry.level1"]
    similarity2 = diagnostics["routing_similarity.chemical__geometry.level2"]
    assert 0.0 <= similarity1 <= 1.0
    assert 0.0 <= similarity2 <= 1.0
    fusion_sum = diagnostics["fusion.chemical"] + diagnostics["fusion.geometry"]
    assert torch.allclose(fusion_sum, torch.tensor(1.0), atol=1.0e-5)
