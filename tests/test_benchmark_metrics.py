import numpy as np
import pandas as pd
import pytest

from benchmarks.metrics import (
    assess_summaries,
    evaluate_fraction_weights,
    evaluate_results,
    summarize_replicates,
)


def test_metrics_count_missing_results_as_nonfinite_and_nondiscoveries() -> None:
    truth = pd.DataFrame({
        'geneId': ['a', 'b', 'c', 'd'],
        'meanInputCount': [100, 100, 100, 100],
        'effect': [0, 0, 1, -1],
    })
    results = pd.DataFrame({
        'geneId': ['a', 'b', 'c'],
        'estimate': [0.1, 1.0, 1.2],
        'standardError': [0.2, 0.2, 0.2],
        'pValue': [0.5, 0.01, 0.001],
        'adjustedPValue': [0.5, 0.04, 0.004],
    })

    metrics = evaluate_results(results, truth, truth_column = 'effect')

    assert metrics.tested_genes == 4
    assert metrics.finite_rate == 0.75
    assert metrics.null_finite_rate == 1
    assert metrics.alternative_finite_rate == 0.5
    assert metrics.false_discoveries == 1
    assert metrics.true_discoveries == 1
    assert metrics.false_discovery_rate == 0.5
    assert metrics.power == 0.5
    assert metrics.positive_alternative_genes == 1
    assert metrics.negative_alternative_genes == 1
    assert metrics.positive_power == 1
    assert metrics.negative_power == 0


def test_fraction_metrics_align_rows_and_reject_duplicates() -> None:
    truth = pd.DataFrame({
        'participantId': ['a', 'a', 'a'],
        'condition': ['Control'] * 3,
        'fraction': ['monosome', 'polysome', 'lost'],
        'trueFraction': [0.3, 0.6, 0.1],
    })
    estimates = truth.rename(columns = {'trueFraction': 'estimatedFraction'})

    observed = evaluate_fraction_weights(estimates, truth)
    assert observed['fractionFiniteRate'] == 1
    assert observed['fractionRmse'] == pytest.approx(0)
    assert observed['fractionOutOfRangeRate'] == 0
    assert observed['fractionSimplexMaxAbsoluteError'] == pytest.approx(0)
    assert observed['monosomeBias'] == pytest.approx(0)
    assert observed['polysomeBias'] == pytest.approx(0)
    assert observed['lostBias'] == pytest.approx(0)

    duplicated = pd.concat([estimates, estimates.iloc[[0]]], ignore_index = True)
    with pytest.raises(ValueError, match = 'unique'):
        evaluate_fraction_weights(duplicated, truth)


def test_fraction_metrics_do_not_hide_compensating_bias() -> None:
    truth = pd.DataFrame({
        'participantId': ['a', 'a', 'a', 'b', 'b', 'b'],
        'condition': ['Control'] * 3 + ['Test'] * 3,
        'fraction': ['monosome', 'polysome', 'lost'] * 2,
        'trueFraction': [0.3, 0.6, 0.1, 0.3, 0.6, 0.1],
    })
    estimates = truth.loc[:, ['participantId', 'condition', 'fraction']].copy()
    estimates['estimatedFraction'] = [0.2, 0.5, 0.3, 0.2, 0.5, 0.3]

    observed = evaluate_fraction_weights(estimates, truth)

    assert observed['monosomeBias'] == pytest.approx(-0.1)
    assert observed['polysomeBias'] == pytest.approx(-0.1)
    assert observed['lostBias'] == pytest.approx(0.2)
    assert observed['fractionRmse'] > 0
    assert observed['fractionSimplexMaxAbsoluteError'] == pytest.approx(0)


def test_fraction_metrics_reject_invalid_truth_and_extra_estimates() -> None:
    truth = pd.DataFrame({
        'participantId': ['a', 'a', 'a'],
        'condition': ['Control'] * 3,
        'fraction': ['monosome', 'polysome', 'lost'],
        'trueFraction': [0.3, 0.6, 0.2],
    })
    estimates = truth.rename(columns = {'trueFraction': 'estimatedFraction'})

    with pytest.raises(ValueError, match = 'sum to one'):
        evaluate_fraction_weights(estimates, truth)

    truth['trueFraction'] = [0.3, 0.6, 0.1]
    extra = pd.concat([
        estimates.iloc[:0],
        pd.DataFrame({
            'participantId': ['b'],
            'condition': ['Control'],
            'fraction': ['monosome'],
            'estimatedFraction': [0.3],
        }),
    ])
    with pytest.raises(ValueError, match = 'absent from truth'):
        evaluate_fraction_weights(extra, truth)


def test_replicate_summary_uses_replicates_as_units() -> None:
    records = pd.DataFrame({
        'scenarioId': ['null'] * 2,
        'method': ['model'] * 2,
        'estimand': ['contrast'] * 2,
        'truthColumn': ['effect'] * 2,
        'replicate': [0, 1],
        'seed': [1, 2],
        'runtimeSeconds': [1.0, 3.0],
        'methodError': [None, 'failed'],
        'rejectedGenes': [0, 2],
        'falseDiscoveries': [0, 2],
        'trueDiscoveries': [0, 0],
        'alternativeGenes': [0, 0],
        **{
            name: [np.nan, np.nan]
            for name in (
                'testedGenes',
                'nullGenes',
                'positiveAlternativeGenes',
                'negativeAlternativeGenes',
                'rawFalsePositiveGenes',
                'typeIError',
                'falseDiscoveryRate',
                'power',
                'positivePower',
                'negativePower',
                'effectBias',
                'effectRmse',
                'alternativeEffectBias',
                'alternativeEffectRmse',
                'positiveAlternativeEffectBias',
                'negativeAlternativeEffectBias',
                'intervalCoverage',
                'nullIntervalCoverage',
                'alternativeIntervalCoverage',
                'falseSignRate',
                'finiteRate',
                'nullFiniteRate',
                'alternativeFiniteRate',
            )
        },
    })

    observed = summarize_replicates(records).iloc[0]
    assert observed['replicates'] == 2
    assert observed['completedReplicates'] == 1
    assert observed['failedReplicates'] == 1
    assert observed['anyDiscoveryRate'] == 0.5
    assert observed['pooledFalseDiscoveryRate'] == 1
    assert np.isnan(observed['pooledPower'])
    assert observed['runtimeSecondsMean'] == 2


def test_summary_assessment_applies_prespecified_criteria() -> None:
    summaries = pd.DataFrame({
        'scenarioId': ['pass', 'fail', 'globalNull'],
        'method': ['model', 'model', 'model'],
        'estimand': ['contrast', 'contrast', 'contrast'],
        'failedReplicates': [0, 1, 0],
        'typeIErrorMean': [0.05, 0.08, 0.05],
        'falseDiscoveryRateMean': [0.05, 0.07, 0.05],
        'pooledFalseDiscoveryRate': [0.05, 0.07, 1.0],
        'alternativeGenesMean': [10, 10, 0],
        'intervalCoverageMean': [0.95, np.nan, 0.95],
        'finiteRateMean': [1.0, 0.9, 1.0],
        'positiveAlternativeEffectBiasMean': [0.05, 0.2, np.nan],
        'negativeAlternativeEffectBiasMean': [-0.08, -0.3, np.nan],
    })

    observed = assess_summaries(summaries)

    assert observed.loc[0, 'overallStatus'] == 'pass'
    assert observed.loc[0, 'intervalCoverageStatus'] == 'pass'
    assert observed.loc[1, 'overallStatus'] == 'fail'
    assert observed.loc[1, 'intervalCoverageStatus'] == 'notEvaluated'
    assert observed.loc[1, 'maximumAbsoluteDirectionalEffectBias'] == 0.3
    assert 'methodFailures' in observed.loc[1, 'failedCriteria']
    assert observed.loc[2, 'pooledFalseDiscoveryRateStatus'] == 'notEvaluated'
    assert observed.loc[2, 'overallStatus'] == 'pass'
