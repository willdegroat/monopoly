from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd

from monopoly._design import _canonical_fraction, build_design
from monopoly._normalization import estimate_size_factors
from monopoly._provenance import build_provenance
from monopoly._validation import validate_control_genes
from monopoly.dataset import MonopolyDataset
from monopoly.exceptions import ContrastError, ModelFitError
from monopoly.results import MonopolyResults

_FIT_TYPES = frozenset({'mean', 'parametric'})
_NORMALIZATION_SCOPES = frozenset({'fraction', 'joint'})
_SIZE_FACTOR_FIT_TYPES = frozenset({'iterative', 'poscounts', 'ratio'})
_ALT_HYPOTHESES = frozenset({'greaterAbs', 'lessAbs', 'greater', 'less'})


class MonopolyModel:
    """Paired condition-by-fraction negative-binomial model."""

    def __init__(
        self,
        *,
        dataset: MonopolyDataset,
        reference: str,
        covariates: str | Sequence[str] | None = None,
    ) -> None:
        self._dataset = dataset
        self._design = build_design(
            dataset,
            reference = reference,
            covariates = covariates,
        )
        self._dds: Any | None = None
        self._fit_config: dict[str, Any] | None = None
        self._fit_status: pd.Series | None = None
        self._library_index: pd.MultiIndex | None = None

    def fit(
        self,
        *,
        normalization: str = 'fraction',
        fit_type: str = 'parametric',
        size_factor_fit_type: str = 'ratio',
        control_genes: Collection[str] | None = None,
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        beta_tol: float = 1e-8,
        n_cpus: int | None = 1,
        low_memory: bool = False,
        verbose: bool = False,
    ) -> MonopolyModel:
        """Fit the shared-dispersion model."""
        config = _validate_fit_options(
            genes = self._dataset.genes,
            normalization = normalization,
            fit_type = fit_type,
            size_factor_fit_type = size_factor_fit_type,
            control_genes = control_genes,
            min_mu = min_mu,
            min_disp = min_disp,
            max_disp = max_disp,
            beta_tol = beta_tol,
            n_cpus = n_cpus,
            low_memory = low_memory,
            verbose = verbose,
        )
        if normalization == 'joint':
            return self._fit_backend(config)

        estimated = estimate_size_factors(
            self._dataset,
            scope = 'fraction',
            fit_type = size_factor_fit_type,
            control_genes = control_genes,
            n_cpus = n_cpus,
        )
        config['effectiveSizeFactorFitType'] = dict(
            estimated.effective_fit_types
        )
        return self._fit_backend(config, size_factors = estimated.size_factors)

    def _fit_backend(
        self,
        config: dict[str, Any],
        *,
        size_factors: pd.Series | None = None,
    ) -> MonopolyModel:
        counts, metadata = self._dataset._to_long()
        library_index = counts.index.copy()
        fixed_size_factors = _validate_fixed_size_factors(
            size_factors,
            library_index = library_index,
        )
        sample_ids = pd.Index(
            [f'library{position}' for position in range(len(counts))],
            name = 'libraryId',
        )
        counts.index = sample_ids
        metadata.index = sample_ids
        design_matrix = self._design.full.copy(deep = True)
        design_matrix.index = sample_ids

        try:
            dds = _fit_pydeseq2(
                counts,
                metadata,
                design_matrix = design_matrix,
                config = config,
                size_factors = fixed_size_factors,
            )
        except Exception as error:
            raise ModelFitError(f'Could not fit the model: {error}') from error

        config['effectiveFitType'] = dds.uns['disp_function_type']
        if fixed_size_factors is None:
            config['effectiveSizeFactorFitType'] = _effective_size_factor_fit_type(
                dds,
                requested = config['sizeFactorFitType'],
            )
        elif 'effectiveSizeFactorFitType' not in config:
            raise ModelFitError('Fraction-wise normalization metadata are missing.')
        self._dds = dds
        self._fit_config = config
        self._fit_status = _fit_status(dds, counts = counts)
        self._library_index = library_index
        return self

    def contrast(
        self,
        numerator: str,
        denominator: str,
        *,
        alpha: float = 0.05,
        cooks_filter: bool = False,
        independent_filter: bool = True,
        lfc_null: float = 0.0,
        alt_hypothesis: str | None = None,
    ) -> MonopolyResults:
        """Test a condition difference between two fractions."""
        dds = self._require_fit()
        options = _validate_contrast_options(
            alpha = alpha,
            cooks_filter = cooks_filter,
            independent_filter = independent_filter,
            lfc_null = lfc_null,
            alt_hypothesis = alt_hypothesis,
        )
        numerator = _canonical_fraction(numerator)
        denominator = _canonical_fraction(denominator)
        vector = self._design.contrast(numerator, denominator)

        try:
            stats = _run_pydeseq2_stats(
                dds,
                vector = vector,
                options = options,
                n_cpus = self._fit_config['nCpus'],
                verbose = self._fit_config['verbose'],
            )
        except Exception as error:
            raise ModelFitError(f'Could not compute the contrast: {error}') from error

        contrast = {
            'numerator': numerator,
            'denominator': denominator,
            'positiveEffect': (
                f'{numerator}-to-{denominator} ratio is higher in '
                f'{self._design.test} than {self._design.reference}'
            ),
            **options,
        }
        provenance = build_provenance(
            self._dataset,
            self._design,
            fit = self._fit_config,
            contrast = contrast,
        )
        return MonopolyResults(
            table = _results_table(stats.results_df, dataset = self._dataset),
            diagnostics = _diagnostics_table(
                dds,
                stats.results_df,
                fit_status = self._fit_status,
                independent_filter = independent_filter,
            ),
            provenance = provenance,
        )

    @property
    def design_matrix(self) -> pd.DataFrame:
        """Return a copy of the full model matrix."""
        return self._design.full.copy(deep = True)

    @property
    def size_factors(self) -> pd.Series:
        """Return fitted library size factors."""
        dds = self._require_fit()
        values = dds.obs['size_factors'].to_numpy(copy = True)
        return pd.Series(values, index = self._library_index, name = 'sizeFactor')

    @property
    def dispersions(self) -> pd.Series:
        """Return final gene-wise dispersion estimates."""
        dds = self._require_fit()
        values = dds.var['dispersions'].to_numpy(copy = True)
        return pd.Series(values, index = self._dataset.genes, name = 'dispersion')

    @property
    def normalized_counts(self) -> pd.DataFrame:
        """Return counts divided by fitted size factors."""
        dds = self._require_fit()
        return pd.DataFrame(
            np.array(dds.layers['normed_counts'], copy = True),
            index = self._library_index,
            columns = self._dataset.genes,
        )

    @property
    def fitted_values(self) -> pd.DataFrame:
        """Return fitted mean counts."""
        dds = self._require_fit()
        design = dds.obsm['design_matrix'].to_numpy(dtype = np.float64)
        coefficients = dds.varm['LFC'].to_numpy(dtype = np.float64)
        size_factors = dds.obs['size_factors'].to_numpy(dtype = np.float64)
        values = size_factors[:, None] * np.exp(design @ coefficients.T)
        values = np.maximum(values, self._fit_config['minMu'])
        values[:, self._fit_status.eq('allZero').to_numpy()] = 0.0
        return pd.DataFrame(
            values,
            index = self._library_index,
            columns = self._dataset.genes,
        )

    @property
    def cooks(self) -> pd.DataFrame:
        """Return Cook's distances retained by the fit."""
        dds = self._require_fit()
        if 'cooks' not in dds.layers:
            raise ModelFitError(
                "Cook's distances were discarded because low_memory = True.",
            )
        return pd.DataFrame(
            np.array(dds.layers['cooks'], copy = True),
            index = self._library_index,
            columns = self._dataset.genes,
        )

    @property
    def fit_summary(self) -> dict[str, Any]:
        """Return counts of model fitting outcomes."""
        self._require_fit()
        counts = self._fit_status.value_counts()
        return {
            'totalGenes': len(self._fit_status),
            'fittedGenes': int(counts.get('fitted', 0)),
            'allZeroGenes': int(counts.get('allZero', 0)),
            'failedGenes': int(counts.get('failed', 0)),
            'cooksAvailable': 'cooks' in self._dds.layers,
            'cooksFilteringAvailable': False,
            'outlierRefittingAvailable': False,
            'normalization': self._fit_config['normalization'],
            'dispersionModel': self._fit_config['dispersionModel'],
            'requestedFitType': self._fit_config['fitType'],
            'effectiveFitType': self._fit_config['effectiveFitType'],
            'requestedSizeFactorFitType': self._fit_config['sizeFactorFitType'],
            'effectiveSizeFactorFitType': self._fit_config[
                'effectiveSizeFactorFitType'
            ],
        }

    def _require_fit(self) -> Any:
        if self._dds is None:
            raise ModelFitError('Call fit() before requesting fitted results.')
        return self._dds

    def __repr__(self) -> str:
        state = 'fitted' if self._dds is not None else 'unfitted'
        return (
            'MonopolyModel('
            f'reference = {self._design.reference!r}, '
            f'test = {self._design.test!r}, state = {state!r})'
        )


def _fit_pydeseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    design_matrix: pd.DataFrame,
    config: dict[str, Any],
    size_factors: np.ndarray | None = None,
) -> Any:
    from pydeseq2.dds import DeseqDataSet

    dds = DeseqDataSet(
        counts = counts,
        metadata = metadata,
        design = design_matrix,
        fit_type = config['fitType'],
        size_factors_fit_type = (
            'ratio' if size_factors is not None else config['sizeFactorFitType']
        ),
        control_genes = config['controlGenes'],
        min_mu = config['minMu'],
        min_disp = config['minDisp'],
        max_disp = config['maxDisp'],
        refit_cooks = False,
        beta_tol = config['betaTolerance'],
        n_cpus = config['nCpus'],
        quiet = not config['verbose'],
        low_memory = config['lowMemory'],
    )
    if size_factors is None:
        dds.deseq2()
    else:
        _fit_pydeseq2_with_size_factors(dds, size_factors = size_factors)
    return dds


def _fit_pydeseq2_with_size_factors(
    dds: Any,
    *,
    size_factors: np.ndarray,
) -> None:
    dds.obs['size_factors'] = size_factors
    dds.layers['normed_counts'] = dds.X / size_factors[:, None]
    dds.var['_normed_means'] = dds.layers['normed_counts'].mean(axis = 0)
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    dds.cooks_outlier()


def _run_pydeseq2_stats(
    dds: Any,
    *,
    vector: np.ndarray,
    options: dict[str, Any],
    n_cpus: int | None,
    verbose: bool,
) -> Any:
    from pydeseq2.ds import DeseqStats

    stats = DeseqStats(
        dds,
        contrast = vector,
        alpha = options['alpha'],
        cooks_filter = options['cooksFilter'],
        independent_filter = options['independentFilter'],
        lfc_null = options['lfcNull'],
        alt_hypothesis = options['altHypothesis'],
        quiet = not verbose,
        n_cpus = n_cpus,
    )
    stats.summary()
    return stats


def _validate_fit_options(
    *,
    genes: pd.Index,
    normalization: str,
    fit_type: str,
    size_factor_fit_type: str,
    control_genes: Collection[str] | None,
    min_mu: float,
    min_disp: float,
    max_disp: float,
    beta_tol: float,
    n_cpus: int | None,
    low_memory: bool,
    verbose: bool,
) -> dict[str, Any]:
    if (
        not isinstance(normalization, str)
        or normalization not in _NORMALIZATION_SCOPES
    ):
        expected = ', '.join(sorted(_NORMALIZATION_SCOPES))
        raise ModelFitError(f'normalization must be one of: {expected}.')
    if not isinstance(fit_type, str) or fit_type not in _FIT_TYPES:
        expected = ', '.join(sorted(_FIT_TYPES))
        raise ModelFitError(f'fit_type must be one of: {expected}.')
    if (
        not isinstance(size_factor_fit_type, str)
        or size_factor_fit_type not in _SIZE_FACTOR_FIT_TYPES
    ):
        expected = ', '.join(sorted(_SIZE_FACTOR_FIT_TYPES))
        raise ModelFitError(f'size_factor_fit_type must be one of: {expected}.')

    controls = validate_control_genes(control_genes, genes = genes)
    min_mu = _positive_float(min_mu, name = 'min_mu')
    min_disp = _positive_float(min_disp, name = 'min_disp')
    max_disp = _positive_float(max_disp, name = 'max_disp')
    beta_tol = _positive_float(beta_tol, name = 'beta_tol')
    if max_disp < min_disp:
        raise ModelFitError('max_disp must be greater than or equal to min_disp.')
    n_cpus = _validate_n_cpus(n_cpus)
    _require_bool(low_memory, name = 'low_memory')
    _require_bool(verbose, name = 'verbose')

    return {
        'dispersionModel': 'shared',
        'inferenceModel': 'asymptoticWald',
        'inferenceScope': 'workingModel',
        'normalization': normalization,
        'fitType': fit_type,
        'sizeFactorFitType': size_factor_fit_type,
        'controlGenes': controls,
        'minMu': min_mu,
        'minDisp': min_disp,
        'maxDisp': max_disp,
        'betaTolerance': beta_tol,
        'nCpus': n_cpus,
        'lowMemory': low_memory,
        'verbose': verbose,
        'cooksFiltering': 'unavailableForParticipantFixedEffects',
        'outlierRefitting': 'unavailableForParticipantFixedEffects',
    }


def _validate_contrast_options(
    *,
    alpha: float,
    cooks_filter: bool,
    independent_filter: bool,
    lfc_null: float,
    alt_hypothesis: str | None,
) -> dict[str, Any]:
    alpha = _finite_float(alpha, name = 'alpha', error = ContrastError)
    if not 0 < alpha < 1:
        raise ContrastError('alpha must be between 0 and 1.')
    lfc_null = _finite_float(
        lfc_null,
        name = 'lfc_null',
        error = ContrastError,
    )
    if lfc_null < 0:
        raise ContrastError('lfc_null cannot be negative.')
    if alt_hypothesis is not None and (
        not isinstance(alt_hypothesis, str)
        or alt_hypothesis not in _ALT_HYPOTHESES
    ):
        expected = ', '.join(sorted(_ALT_HYPOTHESES))
        raise ContrastError(f'alt_hypothesis must be None or one of: {expected}.')
    _require_bool(cooks_filter, name = 'cooks_filter', error = ContrastError)
    if cooks_filter:
        raise ContrastError(
            'cooks_filter = True is unavailable with participant fixed effects.',
        )
    _require_bool(
        independent_filter,
        name = 'independent_filter',
        error = ContrastError,
    )
    return {
        'alpha': alpha,
        'cooksFilter': cooks_filter,
        'independentFilter': independent_filter,
        'lfcNull': lfc_null,
        'altHypothesis': alt_hypothesis,
    }


def _fit_status(dds: Any, *, counts: pd.DataFrame) -> pd.Series:
    all_zero = counts.sum(axis = 0).eq(0).to_numpy()
    dispersions = dds.var['dispersions'].to_numpy(dtype = np.float64)
    coefficients = dds.varm['LFC'].to_numpy(dtype = np.float64)
    failed = ~all_zero & (
        ~np.isfinite(dispersions) | ~np.isfinite(coefficients).all(axis = 1)
    )
    status = np.full(len(counts.columns), 'fitted', dtype = object)
    status[all_zero] = 'allZero'
    status[failed] = 'failed'
    return pd.Series(status, index = counts.columns, name = 'fitStatus')


def _effective_size_factor_fit_type(dds: Any, *, requested: str) -> str:
    if requested == 'ratio' and getattr(dds, 'logmeans', None) is None:
        return 'iterative'
    return requested


def _validate_fixed_size_factors(
    size_factors: pd.Series | None,
    *,
    library_index: pd.MultiIndex,
) -> np.ndarray | None:
    if size_factors is None:
        return None
    if not isinstance(size_factors, pd.Series):
        raise ModelFitError('Fixed size factors must be a pandas Series.')
    if size_factors.index.has_duplicates:
        raise ModelFitError('Fixed size-factor library identifiers must be unique.')
    if not size_factors.index.equals(library_index):
        raise ModelFitError('Fixed size factors must align with all modeled libraries.')
    try:
        values = size_factors.to_numpy(dtype = np.float64, copy = True)
    except (TypeError, ValueError) as error:
        raise ModelFitError('Fixed size factors must be numeric.') from error
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ModelFitError('Fixed size factors must be finite and positive.')
    return values


def _results_table(results: pd.DataFrame, *, dataset: MonopolyDataset) -> pd.DataFrame:
    table = results.reindex(dataset.genes).rename(columns = {
        'lfcSE': 'log2FoldChangeSE',
        'stat': 'statistic',
        'pvalue': 'pValue',
        'padj': 'adjustedPValue',
    }).reset_index(names = 'geneId')
    gene_metadata = dataset.gene_metadata
    if gene_metadata is not None and 'geneName' in gene_metadata.columns:
        table.insert(1, 'geneName', gene_metadata['geneName'].to_numpy(copy = True))
    return table


def _diagnostics_table(
    dds: Any,
    results: pd.DataFrame,
    *,
    fit_status: pd.Series,
    independent_filter: bool,
) -> pd.DataFrame:
    results = results.reindex(fit_status.index)
    cooks_outlier = dds.cooks_outlier().reindex(fit_status.index).astype(bool)
    status = fit_status.copy()
    missing_pvalue = results['pvalue'].isna()
    status.loc[status.eq('fitted') & missing_pvalue & cooks_outlier] = 'cooksOutlier'
    status.loc[status.eq('fitted') & missing_pvalue & ~cooks_outlier] = 'failed'
    independent = (
        independent_filter
        & results['pvalue'].notna()
        & results['padj'].isna()
    )

    return pd.DataFrame({
        'geneId': fit_status.index,
        'fitStatus': status.to_numpy(),
        'dispersion': dds.var['dispersions'].to_numpy(copy = True),
        'maxCooks': _max_cooks(dds, genes = fit_status.index),
        'cooksOutlier': cooks_outlier.to_numpy(),
        'independentFilter': independent.to_numpy(),
    })


def _max_cooks(dds: Any, *, genes: pd.Index) -> np.ndarray:
    if 'cooks' not in dds.layers:
        return np.full(len(genes), np.nan)
    return pd.DataFrame(dds.layers['cooks'], columns = genes).max(
        axis = 0,
        skipna = True,
    ).to_numpy()


def _positive_float(value: float, *, name: str) -> float:
    value = _finite_float(value, name = name)
    if value <= 0:
        raise ModelFitError(f'{name} must be greater than zero.')
    return value


def _finite_float(
    value: float,
    *,
    name: str,
    error: type[Exception] = ModelFitError,
) -> float:
    if isinstance(value, bool):
        raise error(f'{name} must be a finite number.')
    try:
        value = float(value)
    except (TypeError, ValueError) as cause:
        raise error(f'{name} must be a finite number.') from cause
    if not np.isfinite(value):
        raise error(f'{name} must be a finite number.')
    return value


def _validate_n_cpus(n_cpus: int | None) -> int | None:
    if n_cpus is None:
        return None
    if isinstance(n_cpus, bool) or not isinstance(n_cpus, int) or n_cpus < 1:
        raise ModelFitError('n_cpus must be None or a positive integer.')
    return n_cpus


def _require_bool(
    value: bool,
    *,
    name: str,
    error: type[Exception] = ModelFitError,
) -> None:
    if not isinstance(value, bool):
        raise error(f'{name} must be True or False.')
