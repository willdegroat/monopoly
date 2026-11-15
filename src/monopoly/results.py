from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from monopoly.exceptions import DataValidationError


class MonopolyResults:
    """Immutable result tables and analysis provenance."""

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        diagnostics: pd.DataFrame,
        provenance: dict[str, Any],
    ) -> None:
        _validate_result_inputs(
            table = table,
            diagnostics = diagnostics,
            provenance = provenance,
        )
        self._table = table.copy(deep = True)
        self._diagnostics = diagnostics.copy(deep = True)
        self._provenance = copy.deepcopy(provenance)

    @property
    def table(self) -> pd.DataFrame:
        """Return a copy of the statistical results."""
        return self._table.copy(deep = True)

    @property
    def diagnostics(self) -> pd.DataFrame:
        """Return a copy of gene-level diagnostics."""
        return self._diagnostics.copy(deep = True)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a copy of the analysis provenance."""
        return copy.deepcopy(self._provenance)

    def write(self, path: str | Path) -> None:
        """Write results, diagnostics, and provenance."""
        path = _validate_output_path(path)
        separator = ',' if path.suffix.lower() == '.csv' else '\t'
        diagnostics_path = path.with_name(
            f'{path.stem}.diagnostics{path.suffix}'
        )
        provenance_path = path.with_name(f'{path.stem}.provenance.json')

        self._table.to_csv(
            path,
            sep = separator,
            index = False,
            na_rep = 'NA',
        )
        self._diagnostics.to_csv(
            diagnostics_path,
            sep = separator,
            index = False,
            na_rep = 'NA',
        )
        provenance_path.write_text(
            json.dumps(self._provenance, indent = 2, allow_nan = False) + '\n',
            encoding = 'utf-8',
        )

    def __repr__(self) -> str:
        rows = len(self._table)
        contrast = self._provenance.get('contrast', {})
        numerator = contrast.get('numerator', '?')
        denominator = contrast.get('denominator', '?')
        return (
            'MonopolyResults('
            f'contrast = {numerator!r} vs {denominator!r}, genes = {rows})'
        )


def _validate_output_path(path: str | Path) -> Path:
    try:
        path = Path(path)
    except TypeError as error:
        raise DataValidationError('path must be a string or pathlib.Path.') from error

    if path.suffix.lower() not in {'.csv', '.tsv'}:
        raise DataValidationError('Result paths must end in .csv or .tsv.')
    if not path.parent.exists():
        raise DataValidationError(
            f'Output directory does not exist: {path.parent}.',
        )
    if not path.parent.is_dir():
        raise DataValidationError(
            f'Output parent is not a directory: {path.parent}.',
        )
    return path


def _validate_result_inputs(
    *,
    table: pd.DataFrame,
    diagnostics: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    for frame, name in ((table, 'table'), (diagnostics, 'diagnostics')):
        if not isinstance(frame, pd.DataFrame):
            raise DataValidationError(f'{name} must be a pandas DataFrame.')
        if 'geneId' not in frame.columns:
            raise DataValidationError(f'{name} must contain a geneId column.')
        identifiers = frame['geneId']
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise DataValidationError(
                f'{name} geneId values must be nonempty strings.',
            )
        if identifiers.duplicated().any():
            raise DataValidationError(f'{name} geneId values must be unique.')

    if not table['geneId'].equals(diagnostics['geneId']):
        raise DataValidationError(
            'table and diagnostics geneId values must have identical order.',
        )
    if not isinstance(provenance, dict):
        raise DataValidationError('provenance must be a dictionary.')
    try:
        json.dumps(provenance, allow_nan = False)
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            'provenance must contain finite, JSON-serializable values.',
        ) from error
