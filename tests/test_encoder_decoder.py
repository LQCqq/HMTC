import torch

from hmtc.config import HMTFConfig, TreeSpec
from hmtc.data import synthetic_batch
from hmtc.models import HMTFModel


def test_explicit_three_module_path_has_sample_specific_roots() -> None:
    config = HMTFConfig(
        num_elements=8,
        max_sites=5,
        hidden_dim=16,
        num_global_tokens=2,
        num_blocks=2,
        num_rbf=4,
        tree_specs=(
            TreeSpec("chemical", "chemical", (3, 2)),
            TreeSpec("geometry", "geometry", (3, 2)),
        ),
    )
    model = HMTFModel(config)
    state = synthetic_batch(2, config.max_sites, config.num_elements)
    output = model(state, torch.tensor([0.3, 0.7]))
    assert hasattr(model, "encoder")
    assert hasattr(model, "latent_core")
    assert hasattr(model, "decoder")
    chemical_roots = output.traces[0].roots["chemical"]
    assert chemical_roots.shape == (2, config.hidden_dim + 1)
    assert not torch.allclose(chemical_roots[0], chemical_roots[1])
    assert model.encoder.factor_indices["chemical"] == (0, 2, 3)
    assert model.encoder.factor_indices["geometry"] == (0, 1, 3)
    expected_mass = state.existence.sum(dim=1)
    chemical_mass = output.traces[0].masses["chemical"][0].sum(dim=1)
    geometry_mass = output.traces[0].masses["geometry"][0].sum(dim=1)
    assert torch.allclose(chemical_mass, expected_mass, atol=1.0e-5)
    assert torch.allclose(geometry_mass, expected_mass, atol=1.0e-5)


def test_arbitrary_lattice_is_canonicalized_without_distance_change() -> None:
    from hmtc.data import CrystalRecord, NormalizationStats, collate_records
    from hmtc.models.geometry import lattice_params_to_matrix

    matrix = torch.tensor([[2.0, 0.4, 0.1], [0.1, 2.5, 0.3], [0.2, 0.0, 3.0]])
    record = CrystalRecord(
        formation_energy_per_atom=-1.0,
        atomic_numbers=torch.tensor([8, 14]),
        frac_coords=torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.4, 0.8]]),
        lattice_matrix=matrix,
    )
    state = collate_records([record], 100, 20, NormalizationStats(0.0, 1.0))
    canonical = lattice_params_to_matrix(state.lattice)[0]
    assert torch.allclose(canonical @ canonical.T, matrix @ matrix.T, atol=1.0e-5)
