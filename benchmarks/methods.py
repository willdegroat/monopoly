from __future__ import annotations

import os
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

from benchmarks.simulate import SimulatedExperiment
from monopoly import MonopolyModel

PYTHON_METHODS = ('monopoly', 'ratioWelch')
R_METHODS = ('deseq2', 'edgeR', 'fracfixr')
METHODS = (*PYTHON_METHODS, *R_METHODS)

METHOD_ESTIMANDS = {
    'monopoly': ('polyVsMono',),
    'ratioWelch': ('polyVsMono',),
    'deseq2': ('polyVsMono',),
    'edgeR': ('polyVsMono',),
    'fracfixr': (
        'polysomeProportionGlm',
        'monosomeProportionGlm',
        'polysomeVsRemainderLogit',
        'monosomeVsRemainderLogit',
        'polysomeVsRemainderWald',
        'monosomeVsRemainderWald',
    ),
}

TRUTH_COLUMNS = {
    ('monopoly', 'polyVsMono'): 'polyVsMonoLog2FoldChange',
    ('ratioWelch', 'polyVsMono'): 'polyVsMonoLog2FoldChange',
    ('deseq2', 'polyVsMono'): 'polyVsMonoLog2FoldChange',
    ('edgeR', 'polyVsMono'): 'polyVsMonoLog2FoldChange',
    ('fracfixr', 'polysomeProportionGlm'): (
        'polysomeProportionLog2FoldChange'
    ),
    ('fracfixr', 'monosomeProportionGlm'): (
        'monosomeProportionLog2FoldChange'
    ),
    ('fracfixr', 'polysomeVsRemainderLogit'): 'polysomeOddsLog2FoldChange',
    ('fracfixr', 'monosomeVsRemainderLogit'): 'monosomeOddsLog2FoldChange',
    ('fracfixr', 'polysomeVsRemainderWald'): 'polysomeOddsLog2FoldChange',
    ('fracfixr', 'monosomeVsRemainderWald'): 'monosomeOddsLog2FoldChange',
}


@dataclass(frozen = True)
class MethodResult:
    """Contain one method and estimand result table."""

    method: str
    estimand: str
    table: pd.DataFrame
    runtime_seconds: float
    error: str | None = None

    @property
    def truth_column(self) -> str:
        """Return the matching simulation-truth column."""
        return TRUTH_COLUMNS[(self.method, self.estimand)]


@dataclass(frozen = True)
class RMethodResults:
    """Contain gene and fraction outputs from R comparators."""

    results: tuple[MethodResult, ...]
    fraction_estimates: pd.DataFrame


def run_python_methods(
    experiment: SimulatedExperiment,
    *,
    methods: tuple[str, ...],
) -> tuple[MethodResult, ...]:
    """Run requested Python methods on one experiment."""
    results = []
    for method in methods:
        if method == 'monopoly':
            results.append(_run_monopoly(experiment))
        elif method == 'ratioWelch':
            results.append(_run_ratio_welch(experiment))
        else:
            raise ValueError(f'Unknown Python method: {method}.')
    return tuple(results)


def run_r_methods(
    experiment: SimulatedExperiment,
    *,
    methods: tuple[str, ...],
    rscript: str | Path,
    work_directory: Path,
) -> RMethodResults:
    """Run requested R comparators through the frozen adapter."""
    if not methods:
        return RMethodResults(results = (), fraction_estimates = pd.DataFrame())
    unknown = set(methods).difference(R_METHODS)
    if unknown:
        raise ValueError(f'Unknown R methods: {sorted(unknown)}.')
    work_directory.mkdir(parents = True, exist_ok = True)
    counts_path = work_directory / 'counts.tsv'
    metadata_path = work_directory / 'metadata.tsv'
    results_path = work_directory / 'results.tsv'
    fractions_path = work_directory / 'fractions.tsv'
    experiment.count_matrix().to_csv(counts_path, sep = '\t')
    experiment.library_metadata.to_csv(metadata_path, sep = '\t', index = False)

    script = Path(__file__).parent / 'r' / 'run_comparators.R'
    command = [
        str(rscript),
        '--vanilla',
        str(script),
        str(counts_path),
        str(metadata_path),
        str(results_path),
        str(fractions_path),
        ','.join(methods),
    ]
    environment = os.environ.copy()
    environment.setdefault('LANG', 'C')
    table = _run_r_command(
        command,
        environment = environment,
        results_path = results_path,
        fractions_path = fractions_path,
    )
    fraction_estimates = _read_optional_r_table(fractions_path)
    expected = {'method', 'estimand', 'runtimeSeconds', 'errorMessage'}
    missing = expected.difference(table.columns)
    if missing:
        raise RuntimeError(f'R output is missing columns: {sorted(missing)}.')
    outputs = []
    for (method, estimand), group in table.groupby(
        ['method', 'estimand'],
        sort = False,
    ):
        errors = group['errorMessage'].dropna().astype(str).unique()
        error = '; '.join(errors) if len(errors) else None
        runtime = float(group['runtimeSeconds'].iloc[0])
        result_table = group.drop(
            columns = [
                'method',
                'estimand',
                'runtimeSeconds',
                'errorMessage',
            ],
        ).reset_index(drop = True)
        outputs.append(MethodResult(
            method = method,
            estimand = estimand,
            table = result_table,
            runtime_seconds = runtime,
            error = error,
        ))
    observed = {result.method for result in outputs}
    missing_methods = set(methods).difference(observed)
    if missing_methods:
        raise RuntimeError(f'R output omitted methods: {sorted(missing_methods)}.')
    return RMethodResults(
        results = tuple(outputs),
        fraction_estimates = fraction_estimates,
    )


def _run_r_command(
    command: list[str],
    *,
    environment: dict[str, str],
    results_path: Path,
    fractions_path: Path,
) -> pd.DataFrame:
    for attempt in range(2):
        results_path.unlink(missing_ok = True)
        fractions_path.unlink(missing_ok = True)
        completed = subprocess.run(
            command,
            check = False,
            capture_output = True,
            text = True,
            env = environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                'R comparators failed: '
                f'{_subprocess_diagnostics(completed)}',
            )
        table = _read_r_results(results_path)
        if table is not None:
            return table
        if attempt == 0:
            warnings.warn(
                'R comparators returned an empty results file; retrying once.',
                RuntimeWarning,
                stacklevel = 2,
            )
    raise RuntimeError(
        'R comparators returned an empty results file after two attempts: '
        f'{_subprocess_diagnostics(completed)}',
    )


def _read_r_results(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(
            path,
            sep = '\t',
            keep_default_na = False,
            na_values = [''],
        )
    except pd.errors.EmptyDataError:
        return None


def _read_optional_r_table(path: Path) -> pd.DataFrame:
    table = _read_r_results(path)
    return pd.DataFrame() if table is None else table


def _subprocess_diagnostics(completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.strip() or '<empty>'
    stderr = completed.stderr.strip() or '<empty>'
    return f'exit code {completed.returncode}; stdout: {stdout}; stderr: {stderr}'


def r_versions(*, rscript: str | Path) -> pd.DataFrame:
    """Return versions of the frozen R comparator dependencies."""
    script = Path(__file__).parent / 'r' / 'run_comparators.R'
    command = [str(rscript), '--vanilla', str(script), '--versions']
    environment = os.environ.copy()
    environment.setdefault('LANG', 'C')
    completed = subprocess.run(
        command,
        check = False,
        capture_output = True,
        text = True,
        env = environment,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f'Could not query R packages: {message}')
    rows = [line.split('\t', maxsplit = 1) for line in completed.stdout.splitlines()]
    rows = [row for row in rows if len(row) == 2]
    return pd.DataFrame(rows, columns = ['package', 'version'])


def _run_monopoly(experiment: SimulatedExperiment) -> MethodResult:
    started = time.perf_counter()
    try:
        model = MonopolyModel(
            dataset = experiment.dataset,
            reference = 'Control',
        ).fit(n_cpus = 1)
        observed = model.contrast(
            'poly',
            'mono',
            cooks_filter = False,
            independent_filter = False,
        ).table
        table = observed.rename(columns = {
            'log2FoldChange': 'estimate',
            'log2FoldChangeSE': 'standardError',
        }).loc[:, [
            'geneId',
            'estimate',
            'standardError',
            'pValue',
            'adjustedPValue',
        ]]
        error = None
    except Exception as exception:
        table = _failed_table(experiment.truth['geneId'])
        error = f'{type(exception).__name__}: {exception}'
    return MethodResult(
        method = 'monopoly',
        estimand = 'polyVsMono',
        table = table,
        runtime_seconds = time.perf_counter() - started,
        error = error,
    )


def _run_ratio_welch(
    experiment: SimulatedExperiment,
) -> MethodResult:
    started = time.perf_counter()
    try:
        monosome = experiment.dataset.counts('monosome').to_numpy(
            dtype = np.float64,
        )
        polysome = experiment.dataset.counts('polysome').to_numpy(
            dtype = np.float64,
        )
        ratio = np.log2((polysome + 0.5) / (monosome + 0.5))
        conditions = experiment.dataset.metadata['condition'].to_numpy()
        control = ratio[conditions == 'Control']
        test = ratio[conditions == 'Test']
        estimate = test.mean(axis = 0) - control.mean(axis = 0)
        standard_error = np.sqrt(
            test.var(axis = 0, ddof = 1) / len(test)
            + control.var(axis = 0, ddof = 1) / len(control),
        )
        _, p_value = ttest_ind(
            test,
            control,
            axis = 0,
            equal_var = False,
            nan_policy = 'propagate',
        )
        adjusted = np.full_like(p_value, np.nan)
        finite = np.isfinite(p_value)
        adjusted[finite] = false_discovery_control(p_value[finite], method = 'bh')
        table = pd.DataFrame({
            'geneId': experiment.truth['geneId'],
            'estimate': estimate,
            'standardError': standard_error,
            'pValue': p_value,
            'adjustedPValue': adjusted,
        })
        error = None
    except Exception as exception:
        table = _failed_table(experiment.truth['geneId'])
        error = f'{type(exception).__name__}: {exception}'
    return MethodResult(
        method = 'ratioWelch',
        estimand = 'polyVsMono',
        table = table,
        runtime_seconds = time.perf_counter() - started,
        error = error,
    )


def _failed_table(genes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'geneId': genes,
        'estimate': np.nan,
        'standardError': np.nan,
        'pValue': np.nan,
        'adjustedPValue': np.nan,
    })
