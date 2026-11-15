import json

import numpy as np
import pandas as pd
import pytest

from monopoly import (
    ContrastError,
    ModelFitError,
    MonopolyDataset,
    MonopolyModel,
    MonopolyResults,
)
from monopoly.model import (
    _validate_contrast_options,
    _validate_fit_options,
    _validate_fixed_size_factors,
)


@pytest.fixture(scope = 'module')
def modeled_dataset() -> MonopolyDataset:
    rng = np.random.default_rng(3829)
    participants = pd.Index(
        [f'control{number}' for number in range(4)]
        + [f'test{number}' for number in range(4)],
    )
    conditions = np.repeat(['Control', 'Test'], 4)
    genes = pd.Index([f'gene{number}' for number in range(80)] + ['geneZero'])
    baseline = rng.uniform(3.5, 5.5, size = 80)
    participant_effect = rng.normal(0, 0.15, size = len(participants))

    matrices = {}
    for fraction, fraction_effect in {
        'input': 0.0,
        'monosome': -0.35,
        'polysome': 0.45,
    }.items():
        interaction = np.zeros(80)
        if fraction == 'polysome':
            interaction[:12] = 1.2
        log_mean = (
            baseline[None, :]
            + participant_effect[:, None]
            + fraction_effect
            + (conditions == 'Test')[:, None] * interaction[None, :]
        )
        mean = np.exp(log_mean)
        gamma_rate = rng.gamma(shape = 5.0, scale = mean / 5.0)
        counts = rng.poisson(gamma_rate)
        matrices[fraction] = pd.DataFrame(
            np.column_stack([counts, np.zeros(len(participants), dtype = int)]),
            index = participants,
            columns = genes,
        )

    metadata = pd.DataFrame({'condition': conditions}, index = participants)
    gene_metadata = pd.DataFrame(
        {'geneName': [f'Gene {number}' for number in range(len(genes))]},
        index = genes,
    )
    return MonopolyDataset(
        **matrices,
        metadata = metadata,
        gene_metadata = gene_metadata,
    )


@pytest.fixture(scope = 'module')
def fitted_model(modeled_dataset: MonopolyDataset) -> MonopolyModel:
    return MonopolyModel(
        dataset = modeled_dataset,
        reference = 'Control',
    ).fit(n_cpus = 1)


@pytest.fixture(scope = 'module')
def reversed_reference_model(modeled_dataset: MonopolyDataset) -> MonopolyModel:
    return MonopolyModel(
        dataset = modeled_dataset,
        reference = 'Test',
    ).fit(n_cpus = 1)


def test_shared_model_fits_and_exposes_aligned_outputs(
    fitted_model: MonopolyModel,
    modeled_dataset: MonopolyDataset,
) -> None:
    assert "state = 'fitted'" in repr(fitted_model)
    assert fitted_model.design_matrix.shape == (24, 12)
    assert fitted_model.size_factors.index.names == ['participantId', 'fraction']
    assert fitted_model.size_factors.name == 'sizeFactor'
    assert fitted_model.dispersions.index.equals(modeled_dataset.genes)
    assert fitted_model.normalized_counts.shape == (24, 81)
    assert fitted_model.fitted_values.shape == (24, 81)
    assert fitted_model.cooks.shape == (24, 81)
    assert fitted_model.fit_summary == {
        'totalGenes': 81,
        'fittedGenes': 80,
        'allZeroGenes': 1,
        'failedGenes': 0,
        'cooksAvailable': True,
        'cooksFilteringAvailable': False,
        'outlierRefittingAvailable': False,
        'normalization': 'fraction',
        'dispersionModel': 'shared',
        'requestedFitType': 'parametric',
        'effectiveFitType': fitted_model._dds.uns['disp_function_type'],
        'requestedSizeFactorFitType': 'ratio',
        'effectiveSizeFactorFitType': {
            'input': 'ratio',
            'monosome': 'ratio',
            'polysome': 'ratio',
        },
    }

    long_counts, _ = modeled_dataset._to_long()
    expected = long_counts.div(fitted_model.size_factors, axis = 0)
    pd.testing.assert_frame_equal(fitted_model.normalized_counts, expected)
    assert (fitted_model.fitted_values.iloc[:, :-1] >= 0.5).to_numpy().all()
    assert fitted_model.fitted_values['geneZero'].eq(0).all()


def test_model_outputs_are_defensive_copies(fitted_model: MonopolyModel) -> None:
    design = fitted_model.design_matrix
    size_factors = fitted_model.size_factors
    dispersions = fitted_model.dispersions
    design.iloc[0, 0] = -1
    size_factors.iloc[0] = -1
    dispersions.iloc[0] = -1

    assert fitted_model.design_matrix.iloc[0, 0] == 1
    assert fitted_model.size_factors.iloc[0] > 0
    assert fitted_model.dispersions.iloc[0] > 0


def test_joint_normalization_is_an_explicit_alternative(
    modeled_dataset: MonopolyDataset,
) -> None:
    model = MonopolyModel(
        dataset = modeled_dataset,
        reference = 'Control',
    ).fit(normalization = 'joint', n_cpus = 1)

    assert model.fit_summary['normalization'] == 'joint'
    assert model.fit_summary['effectiveSizeFactorFitType'] == 'ratio'


def test_fraction_normalization_stabilizes_fraction_wide_scaling(
    modeled_dataset: MonopolyDataset,
    fitted_model: MonopolyModel,
) -> None:
    scaled = MonopolyDataset(
        input = modeled_dataset.counts('input') * 2,
        monosome = modeled_dataset.counts('monosome') * 5,
        polysome = modeled_dataset.counts('polysome') * 11,
        metadata = modeled_dataset.metadata,
        gene_metadata = modeled_dataset.gene_metadata,
    )
    scaled_model = MonopolyModel(
        dataset = scaled,
        reference = 'Control',
    ).fit(n_cpus = 1)
    original = fitted_model.contrast(
        'poly',
        'mono',
        independent_filter = False,
    ).table
    observed = scaled_model.contrast(
        'poly',
        'mono',
        independent_filter = False,
    ).table
    fitted = original['log2FoldChange'].notna()

    difference = (
        observed.loc[fitted, 'log2FoldChange']
        - original.loc[fitted, 'log2FoldChange']
    ).abs()
    correlation = np.corrcoef(
        observed.loc[fitted, 'log2FoldChange'],
        original.loc[fitted, 'log2FoldChange'],
    )[0, 1]

    # Scaling count depth changes Poisson information, so exact equality is not
    # expected even though fraction constants cancel from the estimand.
    assert difference.median() < 0.01
    assert difference.max() < 0.1
    assert correlation > 0.999


def test_fixed_size_factors_are_validated_without_coercing_alignment(
    modeled_dataset: MonopolyDataset,
) -> None:
    counts, _ = modeled_dataset._to_long()
    factors = pd.Series(1.0, index = counts.index)

    values = _validate_fixed_size_factors(
        factors,
        library_index = counts.index,
    )
    values[0] = -1

    assert factors.iloc[0] == 1


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ('not_series', 'pandas Series'),
        ('duplicate', 'unique'),
        ('misaligned', 'align'),
        ('nonnumeric', 'numeric'),
        ('zero', 'finite and positive'),
        ('infinite', 'finite and positive'),
    ],
)
def test_fixed_size_factors_reject_invalid_values(
    modeled_dataset: MonopolyDataset,
    mutation: str,
    message: str,
) -> None:
    counts, _ = modeled_dataset._to_long()
    factors: object = pd.Series(1.0, index = counts.index)
    if mutation == 'not_series':
        factors = np.ones(len(counts))
    elif mutation == 'duplicate':
        index = counts.index.to_list()
        index[1] = index[0]
        factors.index = pd.MultiIndex.from_tuples(index, names = counts.index.names)
    elif mutation == 'misaligned':
        factors = factors.iloc[::-1]
    elif mutation == 'nonnumeric':
        factors = factors.astype(object)
        factors.iloc[0] = 'invalid'
    elif mutation == 'zero':
        factors.iloc[0] = 0
    else:
        factors.iloc[0] = np.inf

    with pytest.raises(ModelFitError, match = message):
        _validate_fixed_size_factors(
            factors,
            library_index = counts.index,
        )


def test_pairwise_contrast_has_stable_schema_and_direction(
    fitted_model: MonopolyModel,
) -> None:
    results = fitted_model.contrast('poly', 'mono')

    assert isinstance(results, MonopolyResults)
    assert results.table.columns.tolist() == [
        'geneId',
        'geneName',
        'baseMean',
        'log2FoldChange',
        'log2FoldChangeSE',
        'statistic',
        'pValue',
        'adjustedPValue',
    ]
    assert results.diagnostics.columns.tolist() == [
        'geneId',
        'fitStatus',
        'dispersion',
        'maxCooks',
        'cooksOutlier',
        'independentFilter',
    ]
    assert (results.table.loc[:11, 'log2FoldChange'] > 0).mean() >= 0.9
    assert results.diagnostics.iloc[-1]['fitStatus'] == 'allZero'
    assert results.provenance['contrast']['positiveEffect'] == (
        'polysome-to-monosome ratio is higher in Test than Control'
    )
    assert set(results.provenance['software']) == {
        'formulaic',
        'monopolyPolysome',
        'numpy',
        'pandas',
        'pydeseq2',
        'python',
    }
    assert results.provenance['fit']['outlierRefitting'] == (
        'unavailableForParticipantFixedEffects'
    )
    assert results.provenance['fit']['cooksFiltering'] == (
        'unavailableForParticipantFixedEffects'
    )
    assert results.provenance['fit']['effectiveFitType'] in {'mean', 'parametric'}
    assert results.provenance['fit']['effectiveSizeFactorFitType'] == {
        'input': 'ratio',
        'monosome': 'ratio',
        'polysome': 'ratio',
    }
    assert results.provenance['fit']['inferenceScope'] == 'workingModel'


def test_reversing_contrast_reverses_effect_but_not_pvalues(
    fitted_model: MonopolyModel,
) -> None:
    forward = fitted_model.contrast(
        'poly',
        'mono',
        cooks_filter = False,
        independent_filter = False,
    ).table
    reverse = fitted_model.contrast(
        'mono',
        'poly',
        cooks_filter = False,
        independent_filter = False,
    ).table

    np.testing.assert_allclose(
        forward['log2FoldChange'],
        -reverse['log2FoldChange'],
        equal_nan = True,
    )
    np.testing.assert_allclose(
        forward['statistic'],
        -reverse['statistic'],
        equal_nan = True,
    )
    np.testing.assert_allclose(
        forward['pValue'],
        reverse['pValue'],
        equal_nan = True,
    )


def test_reversing_reference_reverses_effect_but_not_pvalues(
    fitted_model: MonopolyModel,
    reversed_reference_model: MonopolyModel,
) -> None:
    options = {
        'cooks_filter': False,
        'independent_filter': False,
    }
    forward = fitted_model.contrast('poly', 'mono', **options).table
    reverse = reversed_reference_model.contrast('poly', 'mono', **options).table

    np.testing.assert_allclose(
        forward['log2FoldChange'],
        -reverse['log2FoldChange'],
        equal_nan = True,
        rtol = 1e-6,
        atol = 1e-6,
    )
    np.testing.assert_allclose(
        forward['pValue'],
        reverse['pValue'],
        equal_nan = True,
        rtol = 1e-5,
        atol = 1e-8,
    )


def test_result_properties_are_defensive_copies(fitted_model: MonopolyModel) -> None:
    results = fitted_model.contrast('poly', 'input')
    table = results.table
    diagnostics = results.diagnostics
    provenance = results.provenance
    table.loc[0, 'geneId'] = 'changed'
    diagnostics.loc[0, 'fitStatus'] = 'changed'
    provenance['fit']['fitType'] = 'changed'

    assert results.table.loc[0, 'geneId'] == 'gene0'
    assert results.diagnostics.loc[0, 'fitStatus'] == 'fitted'
    assert results.provenance['fit']['fitType'] == 'parametric'


def test_result_write_overwrites_table_and_provenance(
    fitted_model: MonopolyModel,
    tmp_path,
) -> None:
    results = fitted_model.contrast('poly', 'mono')
    output = tmp_path / 'polyVsMono.tsv'
    output.write_text('old', encoding = 'utf-8')
    (tmp_path / 'polyVsMono.provenance.json').write_text(
        'old',
        encoding = 'utf-8',
    )

    assert results.write(output) is None

    observed = pd.read_csv(output, sep = '\t', na_values = 'NA')
    pd.testing.assert_frame_equal(observed, results.table)
    provenance = json.loads(
        (tmp_path / 'polyVsMono.provenance.json').read_text(encoding = 'utf-8'),
    )
    assert provenance == results.provenance
    diagnostics = pd.read_csv(
        tmp_path / 'polyVsMono.diagnostics.tsv',
        sep = '\t',
        na_values = 'NA',
    )
    pd.testing.assert_frame_equal(diagnostics, results.diagnostics)


def test_result_write_supports_csv(fitted_model: MonopolyModel, tmp_path) -> None:
    results = fitted_model.contrast('mono', 'input')
    output = tmp_path / 'monoVsInput.csv'

    results.write(output)

    observed = pd.read_csv(output, na_values = 'NA')
    pd.testing.assert_frame_equal(observed, results.table)
    diagnostics = pd.read_csv(
        tmp_path / 'monoVsInput.diagnostics.csv',
        na_values = 'NA',
    )
    pd.testing.assert_frame_equal(diagnostics, results.diagnostics)


def test_mean_poscounts_low_memory_path(
    modeled_dataset: MonopolyDataset,
) -> None:
    model = MonopolyModel(
        dataset = modeled_dataset,
        reference = 'Control',
    ).fit(
        fit_type = 'mean',
        size_factor_fit_type = 'poscounts',
        control_genes = modeled_dataset.genes[:60],
        n_cpus = 1,
        low_memory = True,
    )

    with pytest.raises(ModelFitError, match = 'low_memory'):
        _ = model.cooks

    results = model.contrast('poly', 'mono')
    assert results.diagnostics['maxCooks'].isna().all()
    assert not model.fit_summary['cooksAvailable']


@pytest.mark.parametrize('name', ['size_factors', 'dispersions', 'normalized_counts'])
def test_fitted_outputs_require_fit(
    modeled_dataset: MonopolyDataset,
    name: str,
) -> None:
    model = MonopolyModel(dataset = modeled_dataset, reference = 'Control')

    with pytest.raises(ModelFitError, match = r'Call fit\(\)'):
        getattr(model, name)


def test_contrast_requires_fit(modeled_dataset: MonopolyDataset) -> None:
    model = MonopolyModel(dataset = modeled_dataset, reference = 'Control')

    with pytest.raises(ModelFitError, match = r'Call fit\(\)'):
        model.contrast('poly', 'mono')


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        ({'normalization': 'invalid'}, 'normalization'),
        ({'normalization': []}, 'normalization'),
        ({'fit_type': 'local'}, 'fit_type'),
        ({'fit_type': []}, 'fit_type'),
        ({'size_factor_fit_type': 'invalid'}, 'size_factor_fit_type'),
        ({'size_factor_fit_type': []}, 'size_factor_fit_type'),
        ({'control_genes': []}, 'cannot be empty'),
        ({'control_genes': 'gene1'}, 'collection'),
        ({'control_genes': ['gene1', 'gene1']}, 'duplicates'),
        ({'control_genes': ['missing']}, 'Unknown'),
        ({'min_mu': 0}, 'min_mu'),
        ({'min_disp': np.inf}, 'min_disp'),
        ({'max_disp': 1e-9}, 'max_disp'),
        ({'beta_tol': 'bad'}, 'beta_tol'),
        ({'n_cpus': 0}, 'n_cpus'),
        ({'n_cpus': True}, 'n_cpus'),
        ({'low_memory': 0}, 'low_memory'),
        ({'verbose': 1}, 'verbose'),
    ],
)
def test_fit_options_reject_invalid_values(updates: dict, message: str) -> None:
    options = {
        'genes': pd.Index(['gene1', 'gene2']),
        'normalization': 'fraction',
        'fit_type': 'parametric',
        'size_factor_fit_type': 'ratio',
        'control_genes': None,
        'min_mu': 0.5,
        'min_disp': 1e-8,
        'max_disp': 10.0,
        'beta_tol': 1e-8,
        'n_cpus': 1,
        'low_memory': False,
        'verbose': False,
    }
    options.update(updates)

    with pytest.raises(ModelFitError, match = message):
        _validate_fit_options(**options)


def test_fit_options_canonicalize_control_gene_order() -> None:
    options = _validate_fit_options(
        genes = pd.Index(['gene1', 'gene2', 'gene3']),
        normalization = 'fraction',
        fit_type = 'mean',
        size_factor_fit_type = 'poscounts',
        control_genes = {'gene3', 'gene1'},
        min_mu = 0.1,
        min_disp = 1e-6,
        max_disp = 2,
        beta_tol = 1e-6,
        n_cpus = None,
        low_memory = True,
        verbose = True,
    )

    assert options['controlGenes'] == ['gene1', 'gene3']


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        ({'alpha': 0}, 'alpha'),
        ({'alpha': np.nan}, 'alpha'),
        ({'lfc_null': -1}, 'lfc_null'),
        ({'alt_hypothesis': 'invalid'}, 'alt_hypothesis'),
        ({'alt_hypothesis': []}, 'alt_hypothesis'),
        ({'cooks_filter': 1}, 'cooks_filter'),
        ({'cooks_filter': True}, 'unavailable'),
        ({'independent_filter': 0}, 'independent_filter'),
    ],
)
def test_contrast_options_reject_invalid_values(
    updates: dict,
    message: str,
) -> None:
    options = {
        'alpha': 0.05,
        'cooks_filter': False,
        'independent_filter': True,
        'lfc_null': 0,
        'alt_hypothesis': None,
    }
    options.update(updates)

    with pytest.raises(ContrastError, match = message):
        _validate_contrast_options(**options)


def test_fit_wraps_backend_failure(
    modeled_dataset: MonopolyDataset,
    monkeypatch,
) -> None:
    model = MonopolyModel(dataset = modeled_dataset, reference = 'Control')

    def fail(*args, **kwargs):
        raise RuntimeError('backend failed')

    monkeypatch.setattr('monopoly.model._fit_pydeseq2', fail)
    with pytest.raises(ModelFitError, match = 'backend failed'):
        model.fit()
    assert 'unfitted' in repr(model)


def test_failed_refit_preserves_previous_fit(
    fitted_model: MonopolyModel,
    monkeypatch,
) -> None:
    original = fitted_model.size_factors

    def fail(*args, **kwargs):
        raise RuntimeError('backend failed')

    monkeypatch.setattr('monopoly.model._fit_pydeseq2', fail)
    with pytest.raises(ModelFitError, match = 'backend failed'):
        fitted_model.fit()

    pd.testing.assert_series_equal(fitted_model.size_factors, original)


def test_contrast_wraps_backend_failure(
    fitted_model: MonopolyModel,
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError('statistics failed')

    monkeypatch.setattr('monopoly.model._run_pydeseq2_stats', fail)
    with pytest.raises(ModelFitError, match = 'statistics failed'):
        fitted_model.contrast('poly', 'mono')
