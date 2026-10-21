import torch

from hmtc.config import DiffusionConfig, HMTFConfig, LossConfig, TreeSpec
from hmtc.data import synthetic_batch
from hmtc.diffusion import JointDiffusion
from hmtc.losses import (
    composition_distribution_loss,
    element_count_loss,
    element_denoising_loss,
    hmtf_loss,
)
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
    total, terms = hmtf_loss(
        output,
        clean,
        LossConfig(),
        element_mask=record.element_mask,
        coordinate_target=record.coordinate_score,
    )
    assert torch.isfinite(total)
    assert "distance" in terms
    assert torch.isfinite(terms["distance"])
    assert "tree_redundancy" in terms
    assert "element_count" in terms
    assert terms["routing_entropy"] >= 0
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None)


def test_element_denoising_loss_ignores_unmasked_sites() -> None:
    target = torch.tensor([[0, 1]])
    active = torch.ones_like(target, dtype=torch.bool)
    element_mask = torch.tensor([[True, False]])
    logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    changed_unmasked = logits.clone()
    changed_unmasked[:, 1] = torch.tensor([5.0, -5.0])
    baseline = element_denoising_loss(logits, target, active, element_mask)
    changed = element_denoising_loss(
        changed_unmasked, target, active, element_mask
    )
    assert torch.allclose(baseline, changed)


def test_element_count_loss_prefers_matching_formula() -> None:
    correct_logits = torch.tensor(
        [[[8.0, -8.0, -8.0, -8.0],
          [8.0, -8.0, -8.0, -8.0],
          [-8.0, 8.0, -8.0, -8.0]]]
    )
    wrong_logits = torch.roll(correct_logits, shifts=2, dims=-1)
    existence = torch.ones(1, 3)
    site_mask = torch.ones(1, 3, dtype=torch.bool)
    target_composition = torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]])
    correct = element_count_loss(
        correct_logits, existence, target_composition, site_mask, max_sites=3
    )
    wrong = element_count_loss(
        wrong_logits, existence, target_composition, site_mask, max_sites=3
    )
    assert correct < 1.0e-4
    assert wrong > correct


def test_composition_loss_detects_formula_and_size_errors() -> None:
    target = torch.tensor([[0.20, 0.10, 0.0, 0.0]])
    formula_error = torch.tensor([[0.0, 0.0, 0.20, 0.10]])
    size_error = 0.5 * target
    exact = composition_distribution_loss(target, target)
    assert exact == 0
    assert composition_distribution_loss(formula_error, target) > exact
    assert composition_distribution_loss(size_error, target) > exact
