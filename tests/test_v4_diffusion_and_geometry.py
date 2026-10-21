"""Regression checks for count conditioning and physically valid interfaces."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from hmtc.checkpoint import validate_checkpoint
from hmtc.config import DiffusionConfig, HMTFConfig, TreeSpec
from hmtc.data import (
    JsonlCrystalDataset,
    NormalizationStats,
    compute_count_prior,
    compute_normalization_stats,
    synthetic_batch,
)
from hmtc.diffusion import HMTFSampler, JointDiffusion
from hmtc.diffusion.schedules import (
    coordinate_sigma,
    cosine_alpha_bar,
    ddim_step,
    wrapped_normal_score,
    wrapped_ve_step,
)
from hmtc.models import HMTFModel
from hmtc.models.geometry import EquivariantCoordinateHead, periodic_radius_graph
from hmtc.models.hmtf import HMTFOutput


def small_config() -> HMTFConfig:
    return HMTFConfig(
        num_elements=5, max_sites=5, hidden_dim=8, num_global_tokens=2,
        num_blocks=1, num_rbf=4, rbf_cutoff=3.0,
        geometry_layers=2, geometry_refinement_layers=1, periodic_image_limit=2,
        tree_specs=(TreeSpec("chemical", "chemical", (3, 2)),
                    TreeSpec("geometry", "geometry", (3, 2))),
    )


def test_count_prior_and_lattice_stats_use_dataset_records() -> None:
    data = JsonlCrystalDataset(Path(__file__).parent / "fixtures/tiny.jsonl")
    prior = compute_count_prior(data, 20)
    expected = torch.bincount(torch.tensor([record.atomic_numbers.numel() for record in data]), minlength=21)[1:].float()
    assert torch.equal(prior, expected / expected.sum())
    stats = compute_normalization_stats(data)
    assert len(stats.lattice_mean) == len(stats.lattice_std) == 6
    values = torch.randn(4, 6)
    assert torch.allclose(stats.normalize_lattice(stats.denormalize_lattice(values)), values, atol=1.0e-5)


def test_terminal_masks_fixed_count_and_neutral_padding() -> None:
    clean = synthetic_batch(3, 5, 5)
    diffusion = JointDiffusion(DiffusionConfig(), 5)
    noisy, record = diffusion.q_sample(clean, torch.ones(3))
    active = clean.physical_site_mask()
    assert torch.equal(noisy.existence, clean.existence)
    assert torch.equal(noisy.num_atoms, clean.num_atoms)
    assert torch.all(noisy.elements[active] == 5)
    assert torch.all(noisy.elements[~active] == 0)
    assert torch.all(noisy.frac_coords[~active] == 0)
    assert not record.existence_mask.any()
    assert record.coordinate_score.abs().max() < 1.0e-6
    for time in (0.0, 0.1, 0.5, 0.9):
        state, _ = diffusion.q_sample(clean, torch.full((3,), time))
        assert torch.equal(state.num_atoms, clean.num_atoms)
        assert torch.equal(state.existence, clean.existence)


def test_wrapped_score_matches_log_density_gradient() -> None:
    clean = torch.tensor([[[0.97, 0.2, 0.7]]], dtype=torch.float64)
    noisy = torch.tensor([[[0.03, 0.9, 0.1]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.3], dtype=torch.float64)
    images = torch.arange(-9, 10, dtype=torch.float64)
    displacement = noisy[..., None] - clean[..., None] + images
    log_density = torch.logsumexp(-0.5 * displacement.square() / sigma.square(), dim=-1).sum()
    expected, = torch.autograd.grad(log_density, noisy)
    actual = wrapped_normal_score(noisy.detach(), clean, sigma)
    assert torch.allclose(actual, expected, atol=1.0e-10)
    clean_mean = wrapped_ve_step(noisy.detach(), actual, sigma, torch.zeros_like(sigma))
    assert torch.isfinite(clean_mean).all()


def test_low_noise_oracle_predictor_returns_clean_coordinates() -> None:
    clean = torch.tensor([[[0.98, 0.2, 0.7]]])
    noisy = torch.remainder(clean + torch.tensor([[[0.01, -0.02, 0.015]]]), 1.0)
    sigma = torch.tensor([0.03])
    score = wrapped_normal_score(noisy, clean, sigma)
    recovered = wrapped_ve_step(noisy, score, sigma, torch.zeros_like(sigma))
    delta = torch.remainder(recovered - clean + 0.5, 1.0) - 0.5
    assert delta.abs().max() < 1.0e-6


def test_gaussian_oracle_ddim_uses_the_same_normalized_lattice() -> None:
    clean = torch.randn(2, 6)
    noise = torch.randn_like(clean)
    time, next_time = torch.full((2,), 0.8), torch.full((2,), 0.4)
    alpha, next_alpha = cosine_alpha_bar(time, 1.15)[:, None], cosine_alpha_bar(next_time, 1.15)[:, None]
    current = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    expected = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
    actual = ddim_step(current, clean, time, next_time, 1.15)
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_radius_graph_contains_self_images_and_reports_image_cap() -> None:
    params = torch.tensor([[0.69314718, 0, 0.69314718, 0, 0, 0.69314718]])
    graph = periodic_radius_graph(torch.zeros(1, 1, 3), params,
                                  torch.ones(1, 1, dtype=torch.bool), 2.5, 2)
    assert graph.distance.numel() == 6
    assert torch.allclose(graph.distance, torch.full_like(graph.distance, 2.0))
    assert not graph.image_limit_hit.any()
    capped = periodic_radius_graph(torch.zeros(1, 1, 3), params,
                                   torch.ones(1, 1, dtype=torch.bool), 8.0, 1)
    assert capped.image_limit_hit.all()


def test_coordinate_vector_head_is_rotation_and_reflection_equivariant() -> None:
    torch.manual_seed(12)
    state = synthetic_batch(1, 4, 5)
    state.lattice = torch.tensor([[1.1, 0.0, 1.1, 0.0, 0.0, 1.1]])
    graph = periodic_radius_graph(state.frac_coords, state.lattice, state.physical_site_mask(), 4.0, 2)
    features = torch.randn(1, 4, 8)
    head = EquivariantCoordinateHead(8, 4, 4.0)
    original = head.cartesian_vector(features, graph)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    for transform in (rotation, torch.diag(torch.tensor([-1.0, 1.0, 1.0]))):
        rotated_graph = replace(graph, displacement=graph.displacement @ transform)
        rotated = head.cartesian_vector(features, rotated_graph)
        assert torch.allclose(rotated, original @ transform, atol=1.0e-6)


def test_full_model_is_origin_invariant_and_padding_isolated() -> None:
    torch.manual_seed(7)
    cfg = small_config()
    model = HMTFModel(cfg).eval()
    state = synthetic_batch(1, cfg.max_sites, cfg.num_elements)
    state.existence[:] = torch.tensor([[1, 1, 1, 0, 0]])
    state.site_mask = state.existence.bool()
    shifted = state.clone()
    shift = torch.tensor([0.17, 0.23, 0.31])
    shifted.frac_coords = torch.remainder(state.frac_coords + shift, 1.0)
    shifted.elements[~state.site_mask] = cfg.num_elements
    time = torch.tensor([0.35])
    with torch.no_grad():
        original = model(state, time)
        translated = model(shifted, time)
    assert torch.allclose(original.energy, translated.energy, atol=3.0e-5)
    assert torch.allclose(original.element_logits[state.site_mask], translated.element_logits[state.site_mask], atol=3.0e-5)
    assert torch.allclose(original.coordinate_score, translated.coordinate_score, atol=3.0e-5)
    delta = torch.remainder(translated.frac_coords - original.frac_coords - shift + 0.5, 1.0) - 0.5
    assert delta[state.site_mask].abs().max() < 3.0e-5
    assert torch.all(translated.frac_coords[~state.site_mask] == 0)


class RecordingModel(nn.Module):
    """A changing categorical predictor exposes accidental re-sampling."""

    def __init__(self, config: HMTFConfig, diffusion_config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.diffusion_config = diffusion_config
        self.normalization = NormalizationStats(0.0, 1.0)
        self.states = []

    def forward(self, state, time, active_trees=None):
        self.states.append(state.clone())
        logits = torch.full((*state.elements.shape, self.config.num_elements), -100.0)
        logits[..., len(self.states) % self.config.num_elements] = 100.0
        sigma = coordinate_sigma(time, self.diffusion_config.coordinate_sigma_min, self.diffusion_config.coordinate_sigma_max)
        return HMTFOutput(
            energy=torch.zeros_like(state.energy), lattice=torch.zeros_like(state.lattice),
            composition=torch.zeros_like(state.composition), element_logits=logits,
            frac_coords=state.frac_coords, existence_logits=torch.zeros_like(state.existence),
            traces=[], fixed_existence=state.existence, coordinate_score=torch.zeros_like(state.frac_coords),
            coordinate_sigma=sigma, geometry_image_limit_hit=torch.zeros(state.batch_size, dtype=torch.bool),
        )


def test_sampler_preserves_counts_and_never_resamples_restored_elements() -> None:
    torch.manual_seed(2)
    cfg, diffusion_config = small_config(), DiffusionConfig()
    model = RecordingModel(cfg, diffusion_config)
    sampler = HMTFSampler(model, cfg, diffusion_config)
    counts = torch.tensor([1, 3, 5])
    result = sampler.sample(3, steps=8, num_atoms=counts)
    for state in model.states + [result]:
        assert torch.equal(state.num_atoms, counts)
        assert torch.equal(state.existence, model.states[0].existence)
    for previous, current in zip(model.states, model.states[1:] + [result]):
        restored = previous.physical_site_mask() & (previous.elements != cfg.num_elements)
        assert torch.equal(previous.elements[restored], current.elements[restored])
    assert not (result.elements[result.physical_site_mask()] == cfg.num_elements).any()
    assert torch.all(result.frac_coords[~result.physical_site_mask()] == 0)


def test_prior_sampling_has_multiple_counts_and_no_implicit_fallback() -> None:
    cfg, diffusion_config = small_config(), DiffusionConfig()
    model = RecordingModel(cfg, diffusion_config)
    sampler = HMTFSampler(model, cfg, diffusion_config, [0.6, 0.4, 0, 0, 0])
    torch.manual_seed(17)
    initial = sampler.initial_state(500, "cpu")
    assert set(initial.num_atoms.tolist()) == {1, 2}
    assert abs(float((initial.num_atoms == 1).float().mean()) - 0.6) < 0.1
    with pytest.raises(ValueError, match="count_prior"):
        HMTFSampler(model, cfg, diffusion_config).sample(2, steps=2)
    with pytest.raises(ValueError, match="integer-valued"):
        sampler.initial_state(2, "cpu", torch.tensor([1.5, 2.0]))


def test_old_checkpoints_are_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="newly trained v4"):
        validate_checkpoint({"format_version": 1})
