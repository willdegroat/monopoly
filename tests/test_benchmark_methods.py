import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.methods import run_python_methods, run_r_methods
from benchmarks.simulate import SimulationConfig, simulate_experiment


@pytest.fixture(scope = 'module')
def benchmark_experiment():
    return simulate_experiment(SimulationConfig(
        scenario_id = 'integration',
        participants_per_condition = 3,
        genes = 120,
        redistributed_genes = 20,
        transcription_genes = 20,
        redistribution_log2_effect = 2,
        dispersion_log_sd = 0.3,
        dispersion_correlation = 1.0,
        seed = 31077,
    ))


def test_python_methods_return_complete_aligned_tables(benchmark_experiment) -> None:
    observed = run_python_methods(
        benchmark_experiment,
        methods = ('monopoly', 'ratioWelch'),
    )

    assert [result.method for result in observed] == ['monopoly', 'ratioWelch']
    for result in observed:
        assert result.error is None
        assert result.table['geneId'].tolist() == benchmark_experiment.truth[
            'geneId'
        ].tolist()
        assert result.table['pValue'].notna().all()


def test_r_methods_retry_an_empty_result(
    benchmark_experiment,
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        results = Path(command[5])
        if calls == 1:
            results.write_text('', encoding = 'utf-8')
        else:
            genes = benchmark_experiment.truth['geneId']
            pd.DataFrame({
                'method': 'deseq2',
                'estimand': 'polyVsMono',
                'geneId': genes,
                'estimate': 0.0,
                'standardError': 1.0,
                'pValue': 0.5,
                'adjustedPValue': 0.5,
                'runtimeSeconds': 1.0,
                'errorMessage': np.nan,
            }).to_csv(results, sep = '\t', index = False)
        return subprocess.CompletedProcess(command, 0, stdout = '', stderr = '')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    with pytest.warns(RuntimeWarning, match = 'retrying once'):
        observed = run_r_methods(
            benchmark_experiment,
            methods = ('deseq2',),
            rscript = 'Rscript',
            work_directory = tmp_path,
        )

    assert calls == 2
    assert len(observed.results) == 1
    assert observed.results[0].error is None


def test_r_methods_reject_repeated_empty_results(
    benchmark_experiment,
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        Path(command[5]).write_text('', encoding = 'utf-8')
        return subprocess.CompletedProcess(command, 0, stdout = '', stderr = '')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    with (
        pytest.warns(RuntimeWarning, match = 'retrying once'),
        pytest.raises(RuntimeError, match = 'after two attempts'),
    ):
        run_r_methods(
            benchmark_experiment,
            methods = ('deseq2',),
            rscript = 'Rscript',
            work_directory = tmp_path,
        )

    assert calls == 2


def test_r_methods_accept_empty_optional_fraction_results(
    benchmark_experiment,
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run(command, **kwargs):
        genes = benchmark_experiment.truth['geneId']
        pd.DataFrame({
            'method': 'deseq2',
            'estimand': 'polyVsMono',
            'geneId': genes,
            'estimate': 0.0,
            'standardError': 1.0,
            'pValue': 0.5,
            'adjustedPValue': 0.5,
            'runtimeSeconds': 1.0,
            'errorMessage': np.nan,
        }).to_csv(Path(command[5]), sep = '\t', index = False)
        Path(command[6]).write_text('\n', encoding = 'utf-8')
        return subprocess.CompletedProcess(command, 0, stdout = '', stderr = '')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    observed = run_r_methods(
        benchmark_experiment,
        methods = ('deseq2',),
        rscript = 'Rscript',
        work_directory = tmp_path,
    )

    assert len(observed.results) == 1
    assert observed.fraction_estimates.empty


@pytest.mark.r_integration
def test_r_methods_run_and_deseq2_matches_monopoly(
    benchmark_experiment,
    tmp_path: Path,
) -> None:
    rscript = os.environ.get('MONOPOLY_RSCRIPT')
    if not rscript:
        pytest.skip('MONOPOLY_RSCRIPT is not configured.')
    python_result = run_python_methods(
        benchmark_experiment,
        methods = ('monopoly',),
    )[0]
    observed = run_r_methods(
        benchmark_experiment,
        methods = ('deseq2', 'edgeR', 'fracfixr'),
        rscript = rscript,
        work_directory = tmp_path,
    )

    assert len(observed.results) == 8
    fracfixr_estimands = {
        result.estimand
        for result in observed.results
        if result.method == 'fracfixr'
    }
    assert fracfixr_estimands == {
        'monosomeProportionGlm',
        'monosomeVsRemainderLogit',
        'monosomeVsRemainderWald',
        'polysomeProportionGlm',
        'polysomeVsRemainderLogit',
        'polysomeVsRemainderWald',
    }
    errors = {
        result.estimand: result.error
        for result in observed.results
        if result.error is not None
    }
    assert errors == {
        'monosomeVsRemainderWald': (
            'FracFixR Wald returned no finite p-values.'
        ),
        'polysomeVsRemainderWald': (
            'FracFixR Wald returned no finite p-values.'
        ),
    }
    deseq2 = next(result for result in observed.results if result.method == 'deseq2')
    difference = (
        deseq2.table.set_index('geneId')['estimate']
        - python_result.table.set_index('geneId')['estimate']
    ).abs()
    assert difference.median() < 2e-3
    assert difference.max() < 0.05
    correlation = np.corrcoef(
        deseq2.table['estimate'],
        python_result.table['estimate'],
    )[0, 1]
    assert correlation > 0.999
    assert len(observed.fraction_estimates) == 18
    sums = observed.fraction_estimates.groupby('participantId')[
        'estimatedFraction'
    ].sum()
    assert sums.to_numpy() == pytest.approx(np.ones(6))
