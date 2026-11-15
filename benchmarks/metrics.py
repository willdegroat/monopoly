from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen = True)
class BenchmarkMetrics:
    """Store gene-level operating characteristics for one replicate."""

    tested_genes: int
    null_genes: int
    alternative_genes: int
    positive_alternative_genes: int
    negative_alternative_genes: int
    rejected_genes: int
    false_discoveries: int
    true_discoveries: int
    raw_false_positive_genes: int
    type_i_error: float
    false_discovery_rate: float
    power: float
    positive_power: float
    negative_power: float
    effect_bias: float
    effect_rmse: float
    alternative_effect_bias: float
    alternative_effect_rmse: float
    positive_alternative_effect_bias: float
    negative_alternative_effect_bias: float
    interval_coverage: float
    null_interval_coverage: float
    alternative_interval_coverage: float
    false_sign_rate: float
    finite_rate: float
    null_finite_rate: float
    alternative_finite_rate: float

    def to_record(self) -> dict[str, int | float]:
        """Return metrics using exported camelCase names."""
        return {
            _camel_case(name): value
            for name, value in asdict(self).items()
        }


def evaluate_results(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_column: str,
    alpha: float = 0.05,
    minimum_mean: float = 20.0,
    confidence: float = 0.95,
) -> BenchmarkMetrics:
    """Evaluate one complete result table against simulation truth."""
    _validate_inputs(
        results,
        truth,
        truth_column = truth_column,
        alpha = alpha,
        minimum_mean = minimum_mean,
        confidence = confidence,
    )
    columns = ['geneId', 'meanInputCount', truth_column]
    merged = truth.loc[:, columns].merge(
        results,
        on = 'geneId',
        how = 'left',
        validate = 'one_to_one',
    )
    expected = merged[truth_column].to_numpy(dtype = np.float64)
    mean_count = merged['meanInputCount'].to_numpy(dtype = np.float64)
    effect = _numeric_column(merged, 'estimate')
    standard_error = _numeric_column(merged, 'standardError')
    p_value = _numeric_column(merged, 'pValue')
    adjusted_p_value = _numeric_column(merged, 'adjustedPValue')

    null = np.isclose(expected, 0, rtol = 0, atol = 1e-12)
    alternative = ~null
    finite = np.isfinite(effect) & np.isfinite(p_value)
    rejected = (
        finite
        & np.isfinite(adjusted_p_value)
        & (adjusted_p_value <= alpha)
    )
    measured = finite & (mean_count >= minimum_mean)
    measured_interval = measured & np.isfinite(standard_error)
    measured_alternative = measured & alternative
    positive_alternative = expected > 1e-12
    negative_alternative = expected < -1e-12
    error = effect[measured] - expected[measured]
    alternative_error = (
        effect[measured_alternative] - expected[measured_alternative]
    )
    positive_error = (
        effect[measured & positive_alternative]
        - expected[measured & positive_alternative]
    )
    negative_error = (
        effect[measured & negative_alternative]
        - expected[measured & negative_alternative]
    )
    critical_value = norm.ppf((1 + confidence) / 2)
    covered = np.zeros(len(merged), dtype = bool)
    covered[measured_interval] = (
        effect[measured_interval]
        - critical_value * standard_error[measured_interval]
        <= expected[measured_interval]
    ) & (
        expected[measured_interval]
        <= effect[measured_interval]
        + critical_value * standard_error[measured_interval]
    )
    rejected_alternative = rejected & alternative & finite

    return BenchmarkMetrics(
        tested_genes = len(merged),
        null_genes = int(null.sum()),
        alternative_genes = int(alternative.sum()),
        positive_alternative_genes = int(positive_alternative.sum()),
        negative_alternative_genes = int(negative_alternative.sum()),
        rejected_genes = int(rejected.sum()),
        false_discoveries = int((rejected & null).sum()),
        true_discoveries = int((rejected & alternative).sum()),
        raw_false_positive_genes = int(
            ((p_value <= alpha) & null & np.isfinite(p_value)).sum(),
        ),
        type_i_error = _rate(p_value[null & np.isfinite(p_value)] <= alpha),
        false_discovery_rate = _ratio(
            int((rejected & null).sum()),
            int(rejected.sum()),
        ),
        power = _rate(rejected[alternative]),
        positive_power = _rate(rejected[positive_alternative]),
        negative_power = _rate(rejected[negative_alternative]),
        effect_bias = _mean(error),
        effect_rmse = _root_mean_square(error),
        alternative_effect_bias = _mean(alternative_error),
        alternative_effect_rmse = _root_mean_square(alternative_error),
        positive_alternative_effect_bias = _mean(positive_error),
        negative_alternative_effect_bias = _mean(negative_error),
        interval_coverage = _rate(covered[measured_interval]),
        null_interval_coverage = _rate(
            covered[null & measured_interval],
        ),
        alternative_interval_coverage = _rate(
            covered[alternative & measured_interval],
        ),
        false_sign_rate = _rate(
            np.sign(effect[rejected_alternative])
            != np.sign(expected[rejected_alternative]),
        ),
        finite_rate = _rate(finite),
        null_finite_rate = _rate(finite[null]),
        alternative_finite_rate = _rate(finite[alternative]),
    )


def evaluate_fraction_weights(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, int | float]:
    """Evaluate reconstructed biological fraction weights."""
    required = {'participantId', 'condition', 'fraction', 'estimatedFraction'}
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f'Fraction estimates are missing: {sorted(missing)}.')
    truth_required = {'participantId', 'condition', 'fraction', 'trueFraction'}
    missing_truth = truth_required.difference(truth.columns)
    if missing_truth:
        raise ValueError(f'Fraction truth is missing: {sorted(missing_truth)}.')
    keys = ['participantId', 'condition', 'fraction']
    if estimates.duplicated(keys).any() or truth.duplicated(keys).any():
        raise ValueError('Fraction rows must be unique by participant and fraction.')
    expected_fractions = {'monosome', 'polysome', 'lost'}
    if set(truth['fraction']) != expected_fractions:
        raise ValueError(
            'Fraction truth must contain monosome, polysome, and lost rows.',
        )
    truth_groups = truth.groupby(['participantId', 'condition'])['fraction'].agg(set)
    if not truth_groups.map(lambda values: values == expected_fractions).all():
        raise ValueError('Each participant requires complete fraction truth.')
    extra = estimates.merge(
        truth.loc[:, keys],
        on = keys,
        how = 'left',
        indicator = True,
    )['_merge'].ne('both')
    if extra.any():
        raise ValueError('Fraction estimates contain rows absent from truth.')
    merged = truth.merge(estimates, on = keys, how = 'left', validate = 'one_to_one')
    observed = _numeric_column(merged, 'estimatedFraction')
    expected = _numeric_column(merged, 'trueFraction')
    if (
        not np.isfinite(expected).all()
        or ((expected < 0) | (expected > 1)).any()
    ):
        raise ValueError('trueFraction must be finite and between zero and one.')
    truth_sums = truth.groupby(['participantId', 'condition'])[
        'trueFraction'
    ].sum()
    if not np.allclose(truth_sums, 1, rtol = 0, atol = 1e-10):
        raise ValueError('Fraction truth must sum to one for each participant.')
    finite = np.isfinite(observed)
    error = observed[finite] - expected[finite]
    participant_keys = ['participantId', 'condition']
    sums = merged.groupby(participant_keys, sort = False)['estimatedFraction'].apply(
        lambda values: values.sum() if values.notna().all() else np.nan,
    )
    record = {
        'fractionRows': len(merged),
        'fractionFiniteRate': _rate(finite),
        'fractionOutOfRangeRate': _rate(
            ((observed < 0) | (observed > 1))[finite],
        ),
        'fractionRmse': _root_mean_square(error),
        'fractionMaxAbsoluteError': (
            float(np.max(np.abs(error))) if len(error) else np.nan
        ),
        'fractionCorrelation': _correlation(
            observed[finite],
            expected[finite],
        ),
        'fractionSimplexMaxAbsoluteError': (
            float(np.max(np.abs(sums.dropna().to_numpy() - 1)))
            if sums.notna().any()
            else np.nan
        ),
    }
    for fraction in ('monosome', 'polysome', 'lost'):
        selected = merged['fraction'].eq(fraction).to_numpy() & finite
        fraction_error = observed[selected] - expected[selected]
        record[f'{fraction}Bias'] = _mean(fraction_error)
        record[f'{fraction}Rmse'] = _root_mean_square(fraction_error)
        record[f'{fraction}Correlation'] = _correlation(
            observed[selected],
            expected[selected],
        )
    return record


def summarize_replicates(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics while retaining replicates as sampling units."""
    required = {
        'scenarioId',
        'method',
        'estimand',
        'replicate',
        'rejectedGenes',
        'falseDiscoveries',
        'trueDiscoveries',
    }
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f'Replicate records are missing: {sorted(missing)}.')
    group_columns = ['scenarioId', 'method', 'estimand', 'truthColumn']
    metric_columns = [
        'runtimeSeconds',
        *(_camel_case(name) for name in BenchmarkMetrics.__dataclass_fields__),
    ]
    summaries = []
    for keys, group in records.groupby(group_columns, dropna = False, sort = True):
        record = dict(zip(group_columns, keys, strict = True))
        record['replicates'] = len(group)
        record['completedReplicates'] = int(group['methodError'].isna().sum())
        record['failedReplicates'] = int(group['methodError'].notna().sum())
        record['anyDiscoveryRate'] = float(group['rejectedGenes'].gt(0).mean())
        record['pooledFalseDiscoveryRate'] = _ratio(
            int(group['falseDiscoveries'].sum()),
            int(group['rejectedGenes'].sum()),
        )
        record['pooledPower'] = _ratio_or_nan(
            int(group['trueDiscoveries'].sum()),
            int(group['alternativeGenes'].sum()),
        )
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors = 'coerce')
            finite = values[np.isfinite(values)]
            record[f'{column}Mean'] = _mean(finite.to_numpy())
            record[f'{column}SE'] = (
                float(finite.std(ddof = 1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else np.nan
            )
        summaries.append(record)
    return pd.DataFrame.from_records(summaries)


def assess_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    """Apply prespecified calibration criteria to benchmark summaries."""
    required = {
        'scenarioId',
        'method',
        'estimand',
        'failedReplicates',
        'typeIErrorMean',
        'falseDiscoveryRateMean',
        'pooledFalseDiscoveryRate',
        'alternativeGenesMean',
        'intervalCoverageMean',
        'finiteRateMean',
        'positiveAlternativeEffectBiasMean',
        'negativeAlternativeEffectBiasMean',
    }
    missing = required.difference(summaries.columns)
    if missing:
        raise ValueError(f'Benchmark summaries are missing: {sorted(missing)}.')
    records = []
    for row in summaries.itertuples(index = False):
        directional_bias = _maximum_absolute_finite([
            row.positiveAlternativeEffectBiasMean,
            row.negativeAlternativeEffectBiasMean,
        ])
        statuses = {
            'methodFailures': (
                'pass' if row.failedReplicates == 0 else 'fail'
            ),
            'typeIError': _upper_status(row.typeIErrorMean, 0.06),
            'meanFalseDiscoveryRate': _upper_status(
                row.falseDiscoveryRateMean,
                0.06,
            ),
            'pooledFalseDiscoveryRate': _upper_status(
                row.pooledFalseDiscoveryRate,
                0.06,
            ) if row.alternativeGenesMean > 0 else 'notEvaluated',
            'intervalCoverage': _range_status(
                row.intervalCoverageMean,
                0.925,
                0.975,
            ),
            'finiteRate': _lower_status(row.finiteRateMean, 0.995),
            'directionalEffectBias': _upper_status(directional_bias, 0.1),
        }
        evaluated = [value for value in statuses.values() if value != 'notEvaluated']
        overall = 'fail' if 'fail' in evaluated else 'pass'
        records.append({
            'scenarioId': row.scenarioId,
            'method': row.method,
            'estimand': row.estimand,
            **{f'{name}Status': value for name, value in statuses.items()},
            'maximumAbsoluteDirectionalEffectBias': directional_bias,
            'overallStatus': overall,
            'failedCriteria': ','.join(
                name for name, value in statuses.items() if value == 'fail'
            ),
        })
    return pd.DataFrame.from_records(records)


def _validate_inputs(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_column: str,
    alpha: float,
    minimum_mean: float,
    confidence: float,
) -> None:
    _validate_identifiers(results, truth, truth_column = truth_column)
    _validate_thresholds(
        alpha = alpha,
        minimum_mean = minimum_mean,
        confidence = confidence,
    )
    _validate_result_values(results)


def _validate_identifiers(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_column: str,
) -> None:
    required = {'geneId', 'estimate', 'standardError', 'pValue', 'adjustedPValue'}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f'Results are missing columns: {sorted(missing)}.')
    truth_required = {'geneId', 'meanInputCount', truth_column}
    missing_truth = truth_required.difference(truth.columns)
    if missing_truth:
        raise ValueError(f'Truth is missing columns: {sorted(missing_truth)}.')
    if results['geneId'].duplicated().any() or truth['geneId'].duplicated().any():
        raise ValueError('Gene identifiers must be unique.')
    extra = set(results['geneId']).difference(truth['geneId'])
    if extra:
        raise ValueError('Results contain genes absent from truth.')


def _validate_thresholds(
    *,
    alpha: float,
    minimum_mean: float,
    confidence: float,
) -> None:
    for value, name in ((alpha, 'alpha'), (confidence, 'confidence')):
        if not _probability(value, closed = False):
            raise ValueError(f'{name} must be between zero and one.')
    if (
        isinstance(minimum_mean, bool)
        or not isinstance(minimum_mean, Real)
        or not np.isfinite(minimum_mean)
        or minimum_mean < 0
    ):
        raise ValueError('minimum_mean must be finite and nonnegative.')


def _validate_result_values(results: pd.DataFrame) -> None:
    for column in ('pValue', 'adjustedPValue'):
        values = _numeric_column(results, column)
        finite = values[np.isfinite(values)]
        if ((finite < 0) | (finite > 1)).any():
            raise ValueError(f'{column} must be between zero and one.')
    standard_errors = _numeric_column(results, 'standardError')
    finite_errors = standard_errors[np.isfinite(standard_errors)]
    if (finite_errors < 0).any():
        raise ValueError('standardError cannot be negative.')


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        return frame[column].to_numpy(dtype = np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{column} must be numeric.') from error


def _probability(value: object, *, closed: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    if not np.isfinite(value):
        return False
    return 0 <= value <= 1 if closed else 0 < value < 1


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) else np.nan


def _rate(values: np.ndarray) -> float:
    return _mean(np.asarray(values, dtype = np.float64))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _ratio_or_nan(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def _correlation(observed: np.ndarray, expected: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) == 0 or np.std(expected) == 0:
        return np.nan
    return float(np.corrcoef(observed, expected)[0, 1])


def _upper_status(value: object, maximum: float) -> str:
    if not _finite_value(value):
        return 'notEvaluated'
    return 'pass' if float(value) <= maximum else 'fail'


def _lower_status(value: object, minimum: float) -> str:
    if not _finite_value(value):
        return 'notEvaluated'
    return 'pass' if float(value) >= minimum else 'fail'


def _range_status(value: object, minimum: float, maximum: float) -> str:
    if not _finite_value(value):
        return 'notEvaluated'
    return 'pass' if minimum <= float(value) <= maximum else 'fail'


def _finite_value(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _maximum_absolute_finite(values: list[object]) -> float:
    finite = [abs(float(value)) for value in values if _finite_value(value)]
    return max(finite) if finite else np.nan


def _camel_case(name: str) -> str:
    parts = name.split('_')
    return ''.join([parts[0], *(part.title() for part in parts[1:])])
