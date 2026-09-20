import torch

from hmtc.config import DiffusionConfig, HMTFConfig, LossConfig, TreeSpec
from hmtc.data import synthetic_batch
from hmtc.diffusion import JointDiffusion
from hmtc.losses import hmtf_loss
from hmtc.models import HMTFModel


def config() -> HMTFConfig:
    return HMTFConfig(
        num_elements=10,
        max_sites=5,
        hidden_dim=12,
        num_global_tokens=2,
        num_blocks=1,
        num_rbf=4,
        tree_specs=(
            TreeSpec("chemical", "chemical", (3, 2)),
            TreeSpec("geometry", "geometry", (3, 2)),
        ),
    )


def test_joint_noise_and_training_loss() -> None:
    model_config = config()
    clean = synthetic_batch(2, model_config.max_sites, model_config.num_elements)
    diffusion = JointDiffusion(DiffusionConfig(), model_config.num_elements)
    time = torch.tensor([0.25, 0.9])
    noisy, record = diffusion.q_sample(clean, time)
    assert noisy.elements.max() <= model_config.num_elements
    assert record.element_mask.shape == clean.elements.shape
    model = HMTFModel(model_config)
    output = model(noisy, time)
    total, terms = hmtf_loss(output, clean, LossConfig())
    assert torch.isfinite(total)
    assert "tree_redundancy" in terms
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
