import numpy as np
import pandas as pd
import pytest

from monopoly import ContrastError, DesignError, MonopolyDataset
from monopoly._design import _materialize, _validate_design_matrix, build_design


def test_design_builds_paired_full_and_reduced_models(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    design = build_design(dataset, reference = 'Control')

    assert design.reference == 'Control'
    assert design.test == 'Test'
    assert design.covariates == ()
    assert design.full.shape == (12, 8)
    assert design.reduced.shape == (12, 6)
    assert design.residual_degrees_of_freedom == 4
    assert design.omnibus_degrees_of_freedom == 2
    assert design.full_formula == (
        '~ participantId + fraction + condition:fraction'
    )
    assert np.linalg.matrix_rank(design.full) == design.full.shape[1]


def test_design_contrasts_encode_difference_of_differences(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)
    design = build_design(dataset, reference = 'Control')
    coefficients = np.zeros(design.full.shape[1])
    coefficient_by_name = dict(
        zip(design.full.columns, range(len(coefficients)), strict = True),
    )
    coefficients[
        coefficient_by_name['fraction[T.monosome]:condition[T.Test]']
    ] = 0.25
    coefficients[
        coefficient_by_name['fraction[T.polysome]:condition[T.Test]']
    ] = 1.25

    poly_vs_mono = design.contrast('poly', 'mono')
    mono_vs_poly = design.contrast('monosome', 'polysome')
    poly_vs_input = design.contrast('polysome', 'input')

    assert poly_vs_mono @ coefficients == pytest.approx(1.0)
    assert mono_vs_poly @ coefficients == pytest.approx(-1.0)
    assert poly_vs_input @ coefficients == pytest.approx(1.25)
    np.testing.assert_array_equal(poly_vs_mono, -mono_vs_poly)


def test_design_contrast_returns_a_copy(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)
    design = build_design(dataset, reference = 'Control')

    contrast = design.contrast('poly', 'mono')
    contrast[:] = 0

    assert np.any(design.contrast('poly', 'mono') != 0)


@pytest.mark.parametrize(
    ('numerator', 'denominator', 'message'),
    [
        ('poly', 'poly', 'different'),
        ('ribosome', 'mono', 'Unknown fraction'),
        ('poly', 'ribosome', 'Unknown fraction'),
    ],
)
def test_design_rejects_invalid_contrasts(
    dataset_inputs: dict[str, pd.DataFrame],
    numerator: str,
    denominator: str,
    message: str,
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)
    design = build_design(dataset, reference = 'Control')

    with pytest.raises(ContrastError, match = message):
        design.contrast(numerator, denominator)


def test_design_supports_balanced_library_covariates(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
) -> None:
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    design = build_design(
        dataset,
        reference = 'Control',
        covariates = 'batch',
    )

    assert design.covariates == ('batch',)
    assert 'batch[T.batch2]' in design.full.columns
    assert design.full.shape == (12, 9)
    assert design.residual_degrees_of_freedom == 3


def test_design_supports_continuous_library_covariates(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
) -> None:
    library_metadata['quality'] = [
        1.0,
        1.4,
        1.1,
        1.3,
        1.2,
        1.8,
        1.7,
        1.5,
        1.9,
        2.0,
        1.6,
        2.2,
    ]
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    design = build_design(
        dataset,
        reference = 'Control',
        covariates = ['quality'],
    )

    assert 'quality' in design.full.columns
    assert np.linalg.matrix_rank(design.full) == design.full.shape[1]


@pytest.mark.parametrize(
    ('values', 'column'),
    [
        ([True, False] * 6, 'flag[T.True]'),
        (['A', 'B'] * 6, 'label[T.B]'),
    ],
)
def test_design_supports_boolean_and_string_covariates(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
    values: list[object],
    column: str,
) -> None:
    covariate = 'flag' if isinstance(values[0], bool) else 'label'
    library_metadata[covariate] = values
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    design = build_design(
        dataset,
        reference = 'Control',
        covariates = covariate,
    )

    assert column in design.full.columns


def test_design_rejects_participant_level_covariates(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    with pytest.raises(DesignError, match = 'absorbed by the participant effect'):
        build_design(dataset, reference = 'Control', covariates = 'age')


def test_design_rejects_covariates_confounded_with_fraction(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
) -> None:
    library_metadata['batch'] = library_metadata['fraction']
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    with pytest.raises(DesignError, match = 'rank deficient'):
        build_design(dataset, reference = 'Control', covariates = 'batch')


@pytest.mark.parametrize(
    ('covariates', 'message'),
    [
        (['batch', 'batch'], 'duplicate'),
        (['fraction'], 'reserved'),
        (['missing'], 'not present'),
        (['invalid-name'], 'Python identifier'),
        (1, 'string or sequence'),
    ],
)
def test_design_rejects_invalid_covariate_requests(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
    covariates: object,
    message: str,
) -> None:
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    with pytest.raises(DesignError, match = message):
        build_design(
            dataset,
            reference = 'Control',
            covariates = covariates,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ('value', 'message'),
    [
        (np.nan, 'missing'),
        (np.inf, 'finite'),
        (1 + 2j, 'complex'),
        ({'invalid': 'mapping'}, 'numeric, boolean, or string'),
    ],
)
def test_design_rejects_invalid_covariate_values(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
    value: object,
    message: str,
) -> None:
    library_metadata['quality'] = pd.Series(
        [value, *range(1, len(library_metadata))],
    )
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    with pytest.raises(DesignError, match = message):
        build_design(dataset, reference = 'Control', covariates = 'quality')


@pytest.mark.parametrize('reference', ['Missing', '', None])
def test_design_rejects_invalid_reference(
    dataset_inputs: dict[str, pd.DataFrame],
    reference: object,
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    with pytest.raises(DesignError, match = 'reference'):
        build_design(dataset, reference = reference)  # type: ignore[arg-type]


def test_design_rejects_non_dataset() -> None:
    with pytest.raises(DesignError, match = 'MonopolyDataset'):
        build_design(object(), reference = 'Control')  # type: ignore[arg-type]


def test_design_is_invariant_to_input_order(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    first = build_design(
        MonopolyDataset(**dataset_inputs),
        reference = 'Control',
    )
    reordered = {
        name: frame.iloc[::-1, ::-1]
        if name != 'metadata'
        else frame.iloc[::-1]
        for name, frame in dataset_inputs.items()
    }
    second = build_design(
        MonopolyDataset(**reordered),
        reference = 'Control',
    )

    np.testing.assert_array_equal(
        first.contrast('poly', 'mono'),
        second.contrast('poly', 'mono'),
    )


def test_design_reports_formula_and_nonfinite_matrix_errors() -> None:
    metadata = pd.DataFrame({'value': [1, 2]})

    with pytest.raises(DesignError, match = 'Could not construct'):
        _materialize('~ missing', metadata)

    matrix = pd.DataFrame({'Intercept': [1.0, np.inf]})
    with pytest.raises(DesignError, match = 'non-finite'):
        _validate_design_matrix(matrix, name = 'Test')


def test_design_rejects_no_residual_degrees_of_freedom() -> None:
    participants = ['controlA', 'controlB', 'testA', 'testB']
    genes = ['geneA']
    counts = pd.DataFrame([[1], [2], [3], [4]], index = participants, columns = genes)
    metadata = pd.DataFrame(
        {'condition': ['Control', 'Control', 'Test', 'Test']},
        index = participants,
    )
    library_metadata = pd.DataFrame({
        'participantId': [
            participant
            for participant in participants
            for _ in range(3)
        ],
        'fraction': ['input', 'monosome', 'polysome'] * 4,
        'covariateA': [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        'covariateB': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        'covariateC': [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        'covariateD': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    })
    dataset = MonopolyDataset(
        input = counts,
        monosome = counts,
        polysome = counts,
        metadata = metadata,
        library_metadata = library_metadata,
    )

    with pytest.raises(DesignError, match = 'residual degrees of freedom'):
        build_design(
            dataset,
            reference = 'Control',
            covariates = [
                'covariateA',
                'covariateB',
                'covariateC',
                'covariateD',
            ],
        )
