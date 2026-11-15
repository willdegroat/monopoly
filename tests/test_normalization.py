import warnings

import numpy as np
import pandas as pd
import pytest

from monopoly import ModelFitError, MonopolyDataset
from monopoly._normalization import estimate_size_factors


@pytest.fixture(scope = 'module')
def scaled_dataset() -> MonopolyDataset:
    participants = pd.Index(['controlA', 'controlB', 'testA', 'testB'])
    genes = pd.Index([f'gene{index}' for index in range(8)])
    baseline = np.arange(10, 90, 10)
    depths = {
        'input': np.array([1, 2, 4, 8]),
        'monosome': np.array([2, 1, 8, 4]),
        'polysome': np.array([4, 8, 1, 2]),
    }
    fraction_scale = {'input': 1, 'monosome': 10, 'polysome': 100}
    matrices = {
        fraction: pd.DataFrame(
            np.outer(depth, baseline * fraction_scale[fraction]),
            index = participants,
            columns = genes,
        )
        for fraction, depth in depths.items()
    }
    metadata = pd.DataFrame(
        {'condition': ['Control', 'Control', 'Test', 'Test']},
        index = participants,
    )
    return MonopolyDataset(**matrices, metadata = metadata)


def test_fraction_normalization_recovers_within_fraction_depths(
    scaled_dataset: MonopolyDataset,
) -> None:
    result = estimate_size_factors(scaled_dataset, scope = 'fraction')

    expected_depths = {
        'input': np.array([1, 2, 4, 8]),
        'monosome': np.array([2, 1, 8, 4]),
        'polysome': np.array([4, 8, 1, 2]),
    }
    for fraction, depths in expected_depths.items():
        observed = result.size_factors.xs(fraction, level = 'fraction')
        expected = depths / np.exp(np.log(depths).mean())
        np.testing.assert_allclose(observed, expected)
        assert np.exp(np.log(observed).mean()) == pytest.approx(1.0)


def test_joint_normalization_includes_fraction_composition(
    scaled_dataset: MonopolyDataset,
) -> None:
    fraction = estimate_size_factors(scaled_dataset, scope = 'fraction')
    joint = estimate_size_factors(scaled_dataset, scope = 'joint')

    fraction_geometric_means = fraction.size_factors.groupby(
        level = 'fraction',
        sort = False,
    ).apply(lambda values: np.exp(np.log(values).mean()))
    joint_geometric_means = joint.size_factors.groupby(
        level = 'fraction',
        sort = False,
    ).apply(lambda values: np.exp(np.log(values).mean()))

    np.testing.assert_allclose(fraction_geometric_means, 1.0)
    np.testing.assert_allclose(
        joint_geometric_means / joint_geometric_means.loc['input'],
        [1.0, 10.0, 100.0],
    )


def test_normalization_supports_control_genes_and_poscounts(
    scaled_dataset: MonopolyDataset,
) -> None:
    result = estimate_size_factors(
        scaled_dataset,
        scope = 'fraction',
        fit_type = 'poscounts',
        control_genes = ['gene0', 'gene1'],
    )

    assert result.scope == 'fraction'
    assert result.fit_type == 'poscounts'
    assert dict(result.effective_fit_types) == {
        'input': 'poscounts',
        'monosome': 'poscounts',
        'polysome': 'poscounts',
    }
    assert np.isfinite(result.size_factors).all()
    assert (result.size_factors > 0).all()


def test_normalization_does_not_mutate_counts(
    scaled_dataset: MonopolyDataset,
) -> None:
    before = scaled_dataset.counts('input')

    estimate_size_factors(scaled_dataset)

    pd.testing.assert_frame_equal(scaled_dataset.counts('input'), before)


@pytest.mark.parametrize(
    ('options', 'message'),
    [
        ({'scope': 'invalid'}, 'scope'),
        ({'scope': []}, 'scope'),
        ({'fit_type': 'invalid'}, 'fit_type'),
        ({'fit_type': []}, 'fit_type'),
        ({'n_cpus': 0}, 'n_cpus'),
        ({'n_cpus': True}, 'n_cpus'),
        ({'control_genes': 'gene0'}, 'collection'),
        ({'control_genes': 1}, 'collection'),
        ({'control_genes': []}, 'empty'),
        ({'control_genes': ['gene0', 'gene0']}, 'duplicates'),
        ({'control_genes': ['missing']}, 'Unknown'),
        ({'control_genes': [1]}, 'nonempty strings'),
    ],
)
def test_normalization_rejects_invalid_options(
    scaled_dataset: MonopolyDataset,
    options: dict,
    message: str,
) -> None:
    with pytest.raises(ModelFitError, match = message):
        estimate_size_factors(scaled_dataset, **options)


def test_fraction_constants_cancel_from_difference_in_differences() -> None:
    means = {
        ('Control', 'monosome'): 20.0,
        ('Control', 'polysome'): 40.0,
        ('Test', 'monosome'): 10.0,
        ('Test', 'polysome'): 80.0,
    }
    constants = {'monosome': 7.0, 'polysome': 0.25}

    def contrast(values):
        return (
            np.log2(values[('Test', 'polysome')])
            - np.log2(values[('Test', 'monosome')])
            - np.log2(values[('Control', 'polysome')])
            + np.log2(values[('Control', 'monosome')])
        )

    rescaled = {
        (condition, fraction): value / constants[fraction]
        for (condition, fraction), value in means.items()
    }

    assert contrast(means) == pytest.approx(contrast(rescaled))


def test_ratio_records_automatic_iterative_fallback() -> None:
    rng = np.random.default_rng(9)
    participants = pd.Index(['controlA', 'controlB', 'testA', 'testB'])
    genes = pd.Index([f'gene{index}' for index in range(40)])
    counts = rng.integers(10, 100, size = (4, 40))
    counts[np.arange(40) % 4, np.arange(40)] = 0
    matrix = pd.DataFrame(counts, index = participants, columns = genes)
    metadata = pd.DataFrame(
        {'condition': ['Control', 'Control', 'Test', 'Test']},
        index = participants,
    )
    dataset = MonopolyDataset(
        input = matrix,
        monosome = matrix,
        polysome = matrix,
        metadata = metadata,
    )

    with warnings.catch_warnings(record = True) as observed_warnings:
        warnings.simplefilter('always')
        result = estimate_size_factors(dataset, fit_type = 'ratio')

    assert any(
        'Switching to iterative' in str(warning.message)
        for warning in observed_warnings
    )
    assert set(result.effective_fit_types.values()) == {'iterative'}
