from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from monopoly._validation import validate_control_genes
from monopoly.dataset import _FRACTIONS, MonopolyDataset
from monopoly.exceptions import ModelFitError

_FIT_TYPES = frozenset({'iterative', 'poscounts', 'ratio'})
_SCOPES = frozenset({'fraction', 'joint'})


@dataclass(frozen = True)
class NormalizationResult:
    size_factors: pd.Series
    scope: str
    fit_type: str
    effective_fit_types: Mapping[str, str]


def estimate_size_factors(
    dataset: MonopolyDataset,
    *,
    scope: str = 'fraction',
    fit_type: str = 'ratio',
    control_genes: Collection[str] | None = None,
    n_cpus: int | None = 1,
) -> NormalizationResult:
    """Estimate joint or fraction-wise DESeq2 size factors."""
    _validate_options(scope = scope, fit_type = fit_type, n_cpus = n_cpus)
    controls = validate_control_genes(
        control_genes,
        genes = dataset.genes,
    )
    counts, metadata = dataset._to_long()

    if scope == 'joint':
        size_factors, effective_fit_type = _estimate_group(
            counts,
            fit_type = fit_type,
            control_genes = controls,
            n_cpus = n_cpus,
        )
        effective_fit_types = {'joint': effective_fit_type}
    else:
        size_factors = pd.Series(index = counts.index, dtype = np.float64)
        effective_fit_types = {}
        for fraction in _FRACTIONS:
            selected = metadata['fraction'].eq(fraction)
            estimates, effective_fit_type = _estimate_group(
                counts.loc[selected],
                fit_type = fit_type,
                control_genes = controls,
                n_cpus = n_cpus,
            )
            size_factors.loc[selected] = estimates.to_numpy()
            effective_fit_types[fraction] = effective_fit_type

    if not np.isfinite(size_factors).all() or (size_factors <= 0).any():
        raise ModelFitError('Normalization produced invalid size factors.')
    size_factors.name = 'sizeFactor'
    return NormalizationResult(
        size_factors = size_factors,
        scope = scope,
        fit_type = fit_type,
        effective_fit_types = MappingProxyType(effective_fit_types),
    )


def _estimate_group(
    counts: pd.DataFrame,
    *,
    fit_type: str,
    control_genes: list[str] | None,
    n_cpus: int | None,
) -> tuple[pd.Series, str]:
    from pydeseq2.dds import DeseqDataSet

    sample_ids = pd.Index(
        [f'library{position}' for position in range(len(counts))],
        name = 'libraryId',
    )
    backend_counts = counts.copy(deep = True)
    backend_counts.index = sample_ids
    metadata = pd.DataFrame(index = sample_ids)
    design = pd.DataFrame({'Intercept': np.ones(len(counts))}, index = sample_ids)

    try:
        dds = DeseqDataSet(
            counts = backend_counts,
            metadata = metadata,
            design = design,
            size_factors_fit_type = fit_type,
            control_genes = control_genes,
            refit_cooks = False,
            n_cpus = n_cpus,
            quiet = True,
        )
        dds.fit_size_factors()
    except Exception as error:
        raise ModelFitError(f'Could not estimate size factors: {error}') from error

    effective_fit_type = fit_type
    if fit_type == 'ratio' and dds.logmeans is None:
        effective_fit_type = 'iterative'
    return (
        pd.Series(
            dds.obs['size_factors'].to_numpy(copy = True),
            index = counts.index,
        ),
        effective_fit_type,
    )


def _validate_options(*, scope: str, fit_type: str, n_cpus: int | None) -> None:
    if not isinstance(scope, str) or scope not in _SCOPES:
        expected = ', '.join(sorted(_SCOPES))
        raise ModelFitError(f'scope must be one of: {expected}.')
    if not isinstance(fit_type, str) or fit_type not in _FIT_TYPES:
        expected = ', '.join(sorted(_FIT_TYPES))
        raise ModelFitError(f'fit_type must be one of: {expected}.')
    if n_cpus is not None and (
        isinstance(n_cpus, bool)
        or not isinstance(n_cpus, int)
        or n_cpus < 1
    ):
        raise ModelFitError('n_cpus must be None or a positive integer.')
