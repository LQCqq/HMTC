import torch

from hmtc.config import HMTFConfig, TreeSpec
from hmtc.data import synthetic_batch
from hmtc.models import HMTFModel


def small_config(extra_tree: bool = False) -> HMTFConfig:
    trees = [
        TreeSpec("chemical", "chemical", (4, 2)),
        TreeSpec("geometry", "geometry", (4, 2)),
    ]
    if extra_tree:
        trees.append(TreeSpec("property", "generic", (3, 2)))
    return HMTFConfig(
        num_elements=12,
        max_sites=6,
        hidden_dim=16,
        num_global_tokens=2,
        num_blocks=2,
        num_rbf=6,
        tree_specs=tuple(trees),
    )


def test_forward_shapes_and_gradients() -> None:
    config = small_config()
    model = HMTFModel(config)
    state = synthetic_batch(2, config.max_sites, config.num_elements)
    output = model(state, torch.tensor([0.2, 0.8]))
    assert output.energy.shape == (2, 1)
    assert output.lattice.shape == (2, 6)
    assert output.element_logits.shape == (2, config.max_sites, config.num_elements)
    assert output.frac_coords.shape == (2, config.max_sites, 3)
    assert set(output.traces[-1].roots) == {"chemical", "geometry"}
    output.energy.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tree_count_is_configuration_driven() -> None:
    config = small_config(extra_tree=True)
    model = HMTFModel(config)
    state = synthetic_batch(1, config.max_sites, config.num_elements)
    output = model(state, torch.tensor([0.5]))
    assert set(output.traces[-1].roots) == {"chemical", "geometry", "property"}
    assert output.traces[-1].fusion_weights.shape[-1] == 3


def test_active_tree_mask() -> None:
    config = small_config(extra_tree=True)
    model = HMTFModel(config)
    state = synthetic_batch(1, config.max_sites, config.num_elements)
    output = model(state, torch.tensor([0.5]), active_trees={"chemical", "property"})
    assert set(output.traces[-1].roots) == {"chemical", "property"}
