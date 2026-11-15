from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from formulaic import model_matrix
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from monopoly.dataset import _FRACTIONS, MonopolyDataset
from monopoly.exceptions import ContrastError, DesignError

_FRACTION_ALIASES = {
    'input': 'input',
    'mono': 'monosome',
    'monosome': 'monosome',
    'poly': 'polysome',
    'polysome': 'polysome',
}
_RESERVED_TERMS = frozenset({'condition', 'fraction', 'participantId'})


@dataclass(frozen = True)
class DesignMatrices:
    full: pd.DataFrame
    reduced: pd.DataFrame
    reference: str
    test: str
    covariates: tuple[str, ...]
    full_formula: str
    reduced_formula: str
    _contrasts: Mapping[tuple[str, str], np.ndarray]

    @property
    def residual_degrees_of_freedom(self) -> int:
        return self.full.shape[0] - self.full.shape[1]

    @property
    def omnibus_degrees_of_freedom(self) -> int:
        return self.full.shape[1] - self.reduced.shape[1]

    def contrast(self, numerator: str, denominator: str) -> np.ndarray:
        numerator = _canonical_fraction(numerator)
        denominator = _canonical_fraction(denominator)
        if numerator == denominator:
            raise ContrastError('Contrast fractions must be different.')

        return self._contrasts[(numerator, denominator)].copy()


def build_design(
    dataset: MonopolyDataset,
    *,
    reference: str,
    covariates: str | Sequence[str] | None = None,
) -> DesignMatrices:
    if not isinstance(dataset, MonopolyDataset):
        raise DesignError('dataset must be a MonopolyDataset.')

    _, metadata = dataset._to_long()
    reference, test = _resolve_conditions(metadata, reference = reference)
    covariates = _validate_covariates(metadata, covariates = covariates)
    metadata = _prepare_metadata(
        metadata,
        participants = dataset.participants,
        reference = reference,
        test = test,
        covariates = covariates,
    )

    reduced_formula = _make_formula([
        'participantId',
        'fraction',
        *covariates,
    ])
    conceptual_full_formula = _make_formula([
        'participantId',
        'fraction',
        *covariates,
        'condition:fraction',
    ])
    expanded_full_formula = _make_formula([
        'participantId',
        'fraction',
        *covariates,
        'condition',
        'fraction:condition',
    ])

    reduced = _materialize(reduced_formula, metadata)
    expanded = _materialize(expanded_full_formula, metadata)
    condition_columns = _term_columns(expanded, term_name = 'condition')
    if len(condition_columns) != 1:  # pragma: no cover
        raise DesignError('The condition main effect could not be identified uniquely.')

    # Participant effects already encode each participant's condition. Removing the
    # condition main effect leaves only estimable condition-by-fraction differences.
    full = expanded.drop(columns = condition_columns)
    _validate_design_matrix(reduced, name = 'Reduced')
    _validate_design_matrix(full, name = 'Full')

    omnibus_df = full.shape[1] - reduced.shape[1]
    if omnibus_df != len(_FRACTIONS) - 1:  # pragma: no cover
        raise DesignError(
            'The condition-by-fraction interaction is not fully estimable.',
        )

    contrasts = _build_contrasts(
        expanded,
        metadata,
        condition_columns = condition_columns,
        reference = reference,
        test = test,
    )

    return DesignMatrices(
        full = full,
        reduced = reduced,
        reference = reference,
        test = test,
        covariates = covariates,
        full_formula = conceptual_full_formula,
        reduced_formula = reduced_formula,
        _contrasts = MappingProxyType(contrasts),
    )


def _resolve_conditions(metadata: pd.DataFrame, *, reference: str) -> tuple[str, str]:
    if not isinstance(reference, str) or not reference.strip():
        raise DesignError('reference must be a nonempty string.')

    conditions = metadata['condition'].drop_duplicates().tolist()
    if reference not in conditions:
        expected = ', '.join(conditions)
        raise DesignError(
            f'Unknown reference {reference!r}. Expected one of: {expected}.',
        )

    test = next(condition for condition in conditions if condition != reference)
    return reference, test


def _validate_covariates(
    metadata: pd.DataFrame,
    *,
    covariates: str | Sequence[str] | None,
) -> tuple[str, ...]:
    covariates = _normalize_covariates(covariates)
    for name in covariates:
        _validate_covariate(metadata, name = name)

    return covariates


def _normalize_covariates(
    covariates: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if covariates is None:
        return ()
    if isinstance(covariates, str):
        covariates = (covariates,)
    else:
        try:
            covariates = tuple(covariates)
        except TypeError as error:
            raise DesignError(
                'covariates must be a string or sequence of strings.',
            ) from error

    if any(not isinstance(name, str) or not name.strip() for name in covariates):
        raise DesignError('covariates must contain nonempty strings.')
    if len(covariates) != len(set(covariates)):
        raise DesignError('covariates cannot contain duplicate names.')

    return tuple(covariates)


def _validate_covariate(metadata: pd.DataFrame, *, name: str) -> None:
    if name in _RESERVED_TERMS:
        raise DesignError(f'{name!r} is a reserved design term.')
    if not name.isidentifier() or keyword.iskeyword(name):
        raise DesignError(
            f'Covariate {name!r} must be a valid Python identifier.',
        )
    if name not in metadata.columns:
        raise DesignError(f'Covariate {name!r} is not present in metadata.')

    values = metadata[name]
    if values.isna().any():
        raise DesignError(f'Covariate {name!r} cannot contain missing values.')
    _validate_covariate_values(values, name = name)
    if _is_participant_level(values, metadata['participantId']):
        raise DesignError(
            f'Covariate {name!r} does not vary within participants and is '
            'already absorbed by the participant effect.',
        )


def _validate_covariate_values(values: pd.Series, *, name: str) -> None:
    if is_complex_dtype(values.dtype):
        raise DesignError(f'Covariate {name!r} cannot contain complex values.')
    if is_numeric_dtype(values.dtype) and not is_bool_dtype(values.dtype):
        numeric = values.to_numpy(dtype = np.float64)
        if not np.isfinite(numeric).all():
            raise DesignError(f'Covariate {name!r} must contain finite values.')
        return

    if is_bool_dtype(values.dtype):
        return
    if all(
        isinstance(value, str) and value.strip()
        for value in values
    ):
        return

    raise DesignError(
        f'Covariate {name!r} must contain numeric, boolean, or string values.',
    )


def _is_participant_level(values: pd.Series, participants: pd.Series) -> bool:
    frame = pd.DataFrame({
        'participantId': participants.to_numpy(),
        'value': values.to_numpy(),
    })
    unique_within_participant = frame.groupby(
        'participantId',
        observed = True,
    )['value'].nunique(dropna = False)
    return bool((unique_within_participant == 1).all())


def _prepare_metadata(
    metadata: pd.DataFrame,
    *,
    participants: pd.Index,
    reference: str,
    test: str,
    covariates: tuple[str, ...],
) -> pd.DataFrame:
    prepared = metadata.loc[
        :,
        ['participantId', 'fraction', 'condition', *covariates],
    ].copy(deep = True)
    prepared['participantId'] = pd.Categorical(
        prepared['participantId'],
        categories = participants,
    )
    prepared['fraction'] = pd.Categorical(
        prepared['fraction'],
        categories = _FRACTIONS,
    )
    prepared['condition'] = pd.Categorical(
        prepared['condition'],
        categories = [reference, test],
    )

    for name in covariates:
        values = prepared[name]
        if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
            prepared[name] = pd.Categorical(values)

    return prepared


def _make_formula(terms: Sequence[str]) -> str:
    return '~ ' + ' + '.join(terms)


def _materialize(formula: str, metadata: pd.DataFrame) -> pd.DataFrame:
    try:
        matrix = model_matrix(formula, metadata, output = 'pandas')
    except Exception as error:
        raise DesignError(f'Could not construct design matrix: {error}') from error

    if matrix.shape[0] != metadata.shape[0]:  # pragma: no cover
        raise DesignError('Design construction unexpectedly removed libraries.')
    return matrix


def _term_columns(matrix: pd.DataFrame, *, term_name: str) -> list[str]:
    term_indices = matrix.model_spec.term_indices
    positions = [
        position
        for term, indices in term_indices.items()
        if str(term) == term_name
        for position in indices
    ]
    return [matrix.columns[position] for position in positions]


def _validate_design_matrix(matrix: pd.DataFrame, *, name: str) -> None:
    values = matrix.to_numpy(dtype = np.float64)
    if not np.isfinite(values).all():
        raise DesignError(f'{name} design matrix contains non-finite values.')

    rank = np.linalg.matrix_rank(values)
    if rank < matrix.shape[1]:
        raise DesignError(
            f'{name} design matrix is rank deficient. Selected covariates are '
            'confounded with participant, fraction, or condition.',
        )
    if matrix.shape[0] - rank <= 0:
        raise DesignError(f'{name} design matrix has no residual degrees of freedom.')


def _build_contrasts(
    expanded: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_columns: list[str],
    reference: str,
    test: str,
) -> dict[tuple[str, str], np.ndarray]:
    contrasts = {}
    for numerator in _FRACTIONS:
        for denominator in _FRACTIONS:
            if numerator == denominator:
                continue
            contrasts[(numerator, denominator)] = _contrast_vector(
                expanded,
                metadata,
                condition_columns = condition_columns,
                numerator = numerator,
                denominator = denominator,
                reference = reference,
                test = test,
            )
    return contrasts


def _contrast_vector(
    expanded: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_columns: list[str],
    numerator: str,
    denominator: str,
    reference: str,
    test: str,
) -> np.ndarray:
    rows = pd.concat([metadata.iloc[[0]]] * 4, ignore_index = True)
    rows.loc[:, 'condition'] = [test, test, reference, reference]
    rows.loc[:, 'fraction'] = [numerator, denominator, numerator, denominator]

    encoded = expanded.model_spec.get_model_matrix(rows)
    encoded = encoded.drop(columns = condition_columns).to_numpy(dtype = np.float64)
    return encoded[0] - encoded[1] - encoded[2] + encoded[3]


def _canonical_fraction(fraction: str) -> str:
    if not isinstance(fraction, str) or fraction not in _FRACTION_ALIASES:
        expected = ', '.join(_FRACTION_ALIASES)
        raise ContrastError(
            f'Unknown fraction {fraction!r}. Expected one of: {expected}.',
        )
    return _FRACTION_ALIASES[fraction]
