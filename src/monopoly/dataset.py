from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from monopoly.exceptions import DataValidationError

_FRACTIONS = ('input', 'monosome', 'polysome')
_LIBRARY_COLUMNS = ('participantId', 'fraction')
_MAX_INT = np.iinfo(np.int64).max
_MAX_FLOAT_COUNT = np.nextafter(np.float64(_MAX_INT), -np.inf)


class MonopolyDataset:
    """Validated count matrices and experiment metadata."""

    def __init__(
        self,
        *,
        input: pd.DataFrame,
        monosome: pd.DataFrame,
        polysome: pd.DataFrame,
        metadata: pd.DataFrame,
        library_metadata: pd.DataFrame | None = None,
        gene_metadata: pd.DataFrame | None = None,
    ) -> None:
        counts = _validate_counts({
            'input': input,
            'monosome': monosome,
            'polysome': polysome,
        })
        participants = counts['input'].index
        genes = counts['input'].columns
        participant_metadata = _validate_metadata(
            metadata,
            participants = participants,
        )

        self._counts = counts
        self._metadata = participant_metadata
        self._library_metadata = _validate_library_metadata(
            library_metadata,
            participants = participants,
            participant_columns = participant_metadata.columns,
        )
        self._gene_metadata = _validate_gene_metadata(
            gene_metadata,
            genes = genes,
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Return a copy of participant metadata."""
        return self._metadata.copy(deep = True)

    @property
    def library_metadata(self) -> pd.DataFrame:
        """Return a copy of library metadata."""
        return self._library_metadata.copy(deep = True)

    @property
    def gene_metadata(self) -> pd.DataFrame | None:
        """Return a copy of gene metadata when available."""
        if self._gene_metadata is None:
            return None

        return self._gene_metadata.copy(deep = True)

    @property
    def participants(self) -> pd.Index:
        """Return participant identifiers in canonical order."""
        return self._counts['input'].index.copy()

    @property
    def genes(self) -> pd.Index:
        """Return gene identifiers in canonical order."""
        return self._counts['input'].columns.copy()

    def counts(self, fraction: str) -> pd.DataFrame:
        """Return a copy of the count matrix for one fraction."""
        if fraction not in _FRACTIONS:
            expected = ', '.join(_FRACTIONS)
            raise DataValidationError(
                f'Unknown fraction {fraction!r}. Expected one of: {expected}.',
            )

        return self._counts[fraction].copy(deep = True)

    def __repr__(self) -> str:
        participants = len(self._counts['input'].index)
        genes = len(self._counts['input'].columns)
        return f'MonopolyDataset(participants = {participants}, genes = {genes})'

    def _to_long(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        library_metadata = self._library_metadata.copy(deep = True)
        library_index = pd.MultiIndex.from_frame(
            library_metadata.loc[:, list(_LIBRARY_COLUMNS)],
        )

        counts = pd.concat(
            self._counts,
            names = ['fraction', 'participantId'],
        ).reorder_levels(['participantId', 'fraction'])
        counts = counts.reindex(library_index)

        participant_metadata = self._metadata.loc[
            library_metadata['participantId']
        ].reset_index(drop = True)
        long_metadata = pd.concat(
            [library_metadata, participant_metadata],
            axis = 1,
        )
        long_metadata.index = library_index

        return counts.copy(deep = True), long_metadata


def _validate_counts(
    matrices: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    validated = {
        fraction: _validate_count_matrix(matrix, name = fraction)
        for fraction, matrix in matrices.items()
    }
    reference = validated['input']

    for fraction in _FRACTIONS[1:]:
        matrix = validated[fraction]
        _require_matching_labels(
            reference.index,
            matrix.index,
            name = f'{fraction} participants',
        )
        _require_matching_labels(
            reference.columns,
            matrix.columns,
            name = f'{fraction} genes',
        )
        validated[fraction] = matrix.loc[reference.index, reference.columns]

    return validated


def _validate_count_matrix(matrix: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise DataValidationError(f'{name} counts must be a pandas DataFrame.')
    if matrix.empty:
        raise DataValidationError(f'{name} counts cannot be empty.')

    _validate_labels(matrix.index, name = f'{name} participant')
    _validate_labels(matrix.columns, name = f'{name} gene')

    _validate_count_dtypes(matrix.dtypes, name = name)
    if matrix.isna().to_numpy().any():
        raise DataValidationError(f'{name} counts cannot contain NA.')

    values = _coerce_count_values(matrix, name = name)
    return pd.DataFrame(
        values,
        index = matrix.index.copy(),
        columns = matrix.columns.copy(),
    )


def _validate_count_dtypes(dtypes: pd.Series, *, name: str) -> None:
    if any(is_bool_dtype(dtype) for dtype in dtypes):
        raise DataValidationError(f'{name} counts cannot contain boolean values.')
    if any(not is_numeric_dtype(dtype) for dtype in dtypes):
        raise DataValidationError(f'{name} counts must be numeric.')
    if any(is_complex_dtype(dtype) for dtype in dtypes):
        raise DataValidationError(f'{name} counts cannot contain complex values.')


def _coerce_count_values(matrix: pd.DataFrame, *, name: str) -> np.ndarray:
    integer_positions = [
        position
        for position, dtype in enumerate(matrix.dtypes)
        if is_integer_dtype(dtype)
    ]
    integer_position_set = set(integer_positions)
    float_positions = [
        position
        for position in range(matrix.shape[1])
        if position not in integer_position_set
    ]
    values = np.empty(matrix.shape, dtype = np.int64)

    if integer_positions:
        values[:, integer_positions] = _coerce_integer_counts(
            matrix.iloc[:, integer_positions],
            name = name,
        )

    if float_positions:
        values[:, float_positions] = _coerce_float_counts(
            matrix.iloc[:, float_positions],
            name = name,
        )

    return values


def _coerce_integer_counts(matrix: pd.DataFrame, *, name: str) -> np.ndarray:
    if (matrix < 0).to_numpy().any():
        raise DataValidationError(f'{name} counts cannot be negative.')
    if (matrix > _MAX_INT).to_numpy().any():
        raise DataValidationError(
            f'{name} counts exceed the supported integer range.',
        )
    return matrix.to_numpy(dtype = np.int64)


def _coerce_float_counts(matrix: pd.DataFrame, *, name: str) -> np.ndarray:
    values = matrix.to_numpy(dtype = np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError(f'{name} counts must be finite.')
    if (values < 0).any():
        raise DataValidationError(f'{name} counts cannot be negative.')
    if not np.equal(values, np.floor(values)).all():
        raise DataValidationError(
            f'{name} counts must contain whole numbers. Round estimated gene counts '
            'once upstream; do not supply TPM or other continuous abundance.',
        )
    if (values > _MAX_FLOAT_COUNT).any():
        raise DataValidationError(
            f'{name} counts exceed the supported integer range.',
        )

    return values.astype(np.int64)


def _validate_metadata(
    metadata: pd.DataFrame,
    *,
    participants: pd.Index,
) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise DataValidationError('metadata must be a pandas DataFrame.')

    _validate_labels(metadata.index, name = 'metadata participant')
    _validate_frame_columns(metadata, name = 'metadata')
    _require_matching_labels(
        participants,
        metadata.index,
        name = 'metadata participants',
    )

    reserved = set(_LIBRARY_COLUMNS).intersection(metadata.columns)
    if reserved:
        columns = ', '.join(sorted(reserved))
        raise DataValidationError(f'metadata contains reserved columns: {columns}.')
    if 'condition' not in metadata.columns:
        raise DataValidationError("metadata must contain a 'condition' column.")

    condition = metadata['condition']
    if condition.isna().any():
        raise DataValidationError('metadata condition values cannot be missing.')
    if any(not isinstance(value, str) or not value.strip() for value in condition):
        raise DataValidationError(
            'metadata condition values must be nonempty strings.',
        )
    if condition.nunique() != 2:
        raise DataValidationError('metadata must contain exactly two condition levels.')

    condition_counts = condition.value_counts()
    under_replicated = condition_counts[condition_counts < 2]
    if not under_replicated.empty:
        levels = ', '.join(under_replicated.index.tolist())
        raise DataValidationError(
            f'Each condition requires at least two participants: {levels}.',
        )

    return metadata.loc[participants].copy(deep = True)


def _validate_library_metadata(
    library_metadata: pd.DataFrame | None,
    *,
    participants: pd.Index,
    participant_columns: pd.Index,
) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [participants, _FRACTIONS],
        names = _LIBRARY_COLUMNS,
    )
    if library_metadata is None:
        return expected.to_frame(index = False)
    if not isinstance(library_metadata, pd.DataFrame):
        raise DataValidationError('library_metadata must be a pandas DataFrame.')

    _validate_frame_columns(library_metadata, name = 'library_metadata')
    missing_columns = set(_LIBRARY_COLUMNS).difference(library_metadata.columns)
    if missing_columns:
        columns = ', '.join(sorted(missing_columns))
        raise DataValidationError(
            f'library_metadata is missing required columns: {columns}.',
        )

    frame = library_metadata.copy(deep = True)
    for column in _LIBRARY_COLUMNS:
        if frame[column].isna().any():
            raise DataValidationError(
                f'library_metadata {column} values cannot be missing.',
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in frame[column]
        ):
            raise DataValidationError(
                f'library_metadata {column} values must be nonempty strings.',
            )

    overlap = set(participant_columns).intersection(frame.columns)
    if overlap:
        columns = ', '.join(sorted(overlap))
        raise DataValidationError(
            f'Participant and library metadata columns overlap: {columns}.',
        )
    if frame.duplicated(list(_LIBRARY_COLUMNS)).any():
        raise DataValidationError(
            'library_metadata contains duplicate participant-fraction rows.',
        )

    observed = pd.MultiIndex.from_frame(frame.loc[:, list(_LIBRARY_COLUMNS)])
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if not missing.empty or not extra.empty:
        details = _format_pair_differences(missing = missing, extra = extra)
        raise DataValidationError(f'library_metadata rows do not match: {details}.')

    indexed = frame.set_index(list(_LIBRARY_COLUMNS), drop = False)
    return indexed.reindex(expected).reset_index(drop = True)


def _validate_gene_metadata(
    gene_metadata: pd.DataFrame | None,
    *,
    genes: pd.Index,
) -> pd.DataFrame | None:
    if gene_metadata is None:
        return None
    if not isinstance(gene_metadata, pd.DataFrame):
        raise DataValidationError('gene_metadata must be a pandas DataFrame.')

    _validate_labels(gene_metadata.index, name = 'gene_metadata gene')
    _validate_frame_columns(gene_metadata, name = 'gene_metadata')
    _require_matching_labels(genes, gene_metadata.index, name = 'gene_metadata genes')

    if 'geneId' in gene_metadata.columns:
        raise DataValidationError(
            "gene_metadata cannot contain the reserved 'geneId' column.",
        )

    return gene_metadata.loc[genes].copy(deep = True)


def _validate_frame_columns(frame: pd.DataFrame, *, name: str) -> None:
    _validate_labels(frame.columns, name = f'{name} column')


def _validate_labels(labels: pd.Index, *, name: str) -> None:
    if not labels.is_unique:
        raise DataValidationError(f'{name} identifiers must be unique.')
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise DataValidationError(f'{name} identifiers must be nonempty strings.')


def _require_matching_labels(
    expected: pd.Index,
    observed: pd.Index,
    *,
    name: str,
) -> None:
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing.empty and extra.empty:
        return

    details = []
    if not missing.empty:
        details.append(f'missing {_preview(missing.tolist())}')
    if not extra.empty:
        details.append(f'unexpected {_preview(extra.tolist())}')

    raise DataValidationError(f'{name} do not match: {"; ".join(details)}.')


def _format_pair_differences(
    *,
    missing: pd.MultiIndex,
    extra: pd.MultiIndex,
) -> str:
    details = []
    if not missing.empty:
        details.append(f'missing {_preview(missing.tolist())}')
    if not extra.empty:
        details.append(f'unexpected {_preview(extra.tolist())}')
    return '; '.join(details)


def _preview(values: list[object], *, limit: int = 5) -> list[object]:
    if len(values) <= limit:
        return values
    return [*values[:limit], '...']
