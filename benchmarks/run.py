from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.methods import (
    METHOD_ESTIMANDS,
    METHODS,
    PYTHON_METHODS,
    R_METHODS,
    MethodResult,
    r_versions,
    run_python_methods,
    run_r_methods,
)
from benchmarks.metrics import (
    assess_summaries,
    evaluate_fraction_weights,
    evaluate_results,
    summarize_replicates,
)
from benchmarks.simulate import (
    SimulatedExperiment,
    SimulationConfig,
    simulate_experiment,
)

FRACTION_METRIC_COLUMNS = [
    'scenarioId',
    'replicate',
    'seed',
    'method',
    'methodError',
    'truthHash',
    'simulationHash',
    'configHash',
    'fractionRows',
    'fractionFiniteRate',
    'fractionOutOfRangeRate',
    'fractionRmse',
    'fractionMaxAbsoluteError',
    'fractionCorrelation',
    'fractionSimplexMaxAbsoluteError',
    'monosomeBias',
    'monosomeRmse',
    'monosomeCorrelation',
    'polysomeBias',
    'polysomeRmse',
    'polysomeCorrelation',
    'lostBias',
    'lostRmse',
    'lostCorrelation',
]

_r_requirements = pd.read_csv(
    Path(__file__).parent / 'r' / 'requirements.tsv',
    sep = '\t',
    dtype = str,
)
R_REQUIREMENTS = dict(
    zip(_r_requirements['package'], _r_requirements['version'], strict = True)
)


@dataclass(frozen = True)
class BenchmarkTask:
    """Define one independently reproducible benchmark replicate."""

    replicate: int
    config: SimulationConfig


@dataclass(frozen = True)
class TaskResult:
    """Contain replicate-level gene and fraction metrics."""

    gene_metrics: pd.DataFrame
    fraction_metrics: pd.DataFrame


TaskArguments = tuple[
    BenchmarkTask,
    tuple[str, ...],
    float,
    float,
    str | None,
    Path,
    str,
]
TaskOutcome = tuple[BenchmarkTask, TaskResult | None, Exception | None]


def run_benchmark(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    rscript: str | Path | None = None,
    workers: int = 1,
) -> None:
    """Run or resume a complete benchmark specification."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    config_path = Path(config_path)
    output_directory = Path(output_directory)
    specification = _read_specification(config_path)
    methods = tuple(specification['methods'])
    tasks = _benchmark_tasks(specification)
    config_hash = _json_hash(specification)
    output_directory.mkdir(parents = True, exist_ok = True)
    parts_directory = output_directory / 'parts'
    work_directory = output_directory / 'work'
    parts_directory.mkdir(exist_ok = True)
    work_directory.mkdir(exist_ok = True)

    versions = _environment_record()
    r_method_names = tuple(method for method in methods if method in R_METHODS)
    if r_method_names:
        if rscript is None:
            raise ValueError('rscript is required for requested R methods.')
        versions['rPackages'] = _r_version_record(rscript = rscript)
        _validate_r_versions(versions['rPackages'], methods = r_method_names)
    else:
        versions['rPackages'] = {}
    manifest = {
        'benchmarkId': specification['benchmarkId'],
        'benchmarkCodeHash': _benchmark_code_hash(),
        'configHash': config_hash,
        'configPath': str(config_path.resolve()),
        'methods': list(methods),
        'replicates': len(tasks),
        'environment': versions,
        'status': 'running',
    }
    _validate_existing_manifest(output_directory / 'manifest.json', manifest)
    _write_json(output_directory / 'manifest.json', manifest)
    _seed_manifest(tasks, config_hash = config_hash).to_csv(
        output_directory / 'seedManifest.tsv',
        sep = '\t',
        index = False,
    )

    pending = [
        task
        for task in tasks
        if not all(path.exists() for path in _part_paths(parts_directory, task))
    ]
    completed = len(tasks) - len(pending)
    progress_interval = max(1, len(tasks) // 100)
    _report_progress(
        completed,
        total = len(tasks),
        benchmark_id = specification['benchmarkId'],
    )
    arguments = [
        (
            task,
            methods,
            float(specification['alpha']),
            float(specification['minimumMean']),
            str(rscript) if rscript is not None else None,
            work_directory,
            config_hash,
        )
        for task in pending
    ]
    failures = []
    for task, result, error in _task_results(arguments, workers = workers):
        failure = _write_task_outcome(
            parts_directory,
            task = task,
            result = result,
            error = error,
        )
        if failure is None:
            completed += 1
        else:
            failures.append((task, failure))
            _report_task_failure(task, failure)
        if completed % progress_interval == 0 or completed == len(tasks):
            _report_progress(
                completed,
                total = len(tasks),
                benchmark_id = specification['benchmarkId'],
            )
    if failures:
        _raise_task_failures(
            failures,
            completed = completed,
            total = len(tasks),
        )

    gene_metrics, fraction_metrics = _collect_parts(
        parts_directory,
        tasks,
        methods = methods,
        config_hash = config_hash,
    )
    gene_metrics.to_csv(
        output_directory / 'replicateResults.tsv',
        sep = '\t',
        index = False,
    )
    summaries = summarize_replicates(gene_metrics)
    summaries.to_csv(
        output_directory / 'metricSummary.tsv',
        sep = '\t',
        index = False,
    )
    assess_summaries(summaries).to_csv(
        output_directory / 'benchmarkAssessment.tsv',
        sep = '\t',
        index = False,
    )
    fraction_metrics.to_csv(
        output_directory / 'fractionRecovery.tsv',
        sep = '\t',
        index = False,
    )
    manifest['status'] = 'complete'
    manifest['resultHash'] = _frame_hash(gene_metrics)
    manifest['scientificResultHash'] = _frame_hash(
        gene_metrics.drop(columns = 'runtimeSeconds'),
    )
    manifest['fractionResultHash'] = _frame_hash(fraction_metrics)
    _write_json(output_directory / 'manifest.json', manifest)


def _task_results(
    arguments: list[TaskArguments],
    *,
    workers: int,
) -> Iterator[TaskOutcome]:
    if workers == 1:
        for argument in arguments:
            task = argument[0]
            try:
                yield task, _run_task(*argument), None
            except Exception as error:
                yield task, None, error
        return
    with ProcessPoolExecutor(max_workers = workers) as executor:
        futures = {
            executor.submit(_run_task, *argument): argument[0]
            for argument in arguments
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                yield task, future.result(), None
            except Exception as error:
                yield task, None, error


def _write_task_outcome(
    directory: Path,
    *,
    task: BenchmarkTask,
    result: TaskResult | None,
    error: Exception | None,
) -> Exception | None:
    if error is not None:
        return error
    if result is None:
        return RuntimeError('Benchmark worker returned no result.')
    try:
        _write_part(directory, task, result)
    except Exception as write_error:
        return write_error
    return None


def _run_task(
    task: BenchmarkTask,
    methods: tuple[str, ...],
    alpha: float,
    minimum_mean: float,
    rscript: str | None,
    work_directory: Path,
    config_hash: str,
) -> TaskResult:
    experiment = simulate_experiment(task.config)
    python_methods = tuple(method for method in methods if method in PYTHON_METHODS)
    method_results = list(run_python_methods(
        experiment,
        methods = python_methods,
    ))
    fraction_estimates = pd.DataFrame()
    r_method_names = tuple(method for method in methods if method in R_METHODS)
    if r_method_names:
        if rscript is None:
            raise ValueError('rscript is required for requested R methods.')
        prefix = f'{task.config.scenario_id}-{task.replicate:04d}-'
        with tempfile.TemporaryDirectory(prefix = prefix, dir = work_directory) as path:
            observed = run_r_methods(
                experiment,
                methods = r_method_names,
                rscript = rscript,
                work_directory = Path(path),
            )
        method_results.extend(observed.results)
        fraction_estimates = observed.fraction_estimates

    truth_hash = _frame_hash(experiment.truth)
    simulation_hash = _simulation_hash(experiment)
    records = [
        _metric_record(
            result,
            experiment = experiment,
            task = task,
            alpha = alpha,
            minimum_mean = minimum_mean,
            truth_hash = truth_hash,
            simulation_hash = simulation_hash,
            config_hash = config_hash,
        )
        for result in method_results
    ]
    fraction_record = pd.DataFrame(columns = FRACTION_METRIC_COLUMNS)
    if 'fracfixr' in methods:
        fraction_error = None
        if fraction_estimates.empty:
            fraction_error = next(
                (
                    result.error
                    for result in method_results
                    if result.method == 'fracfixr' and result.error is not None
                ),
                'FracFixR returned no fraction estimates.',
            )
            fraction_estimates = pd.DataFrame(columns = [
                'participantId',
                'condition',
                'fraction',
                'estimatedFraction',
            ])
        fraction_record = pd.DataFrame([{
            'scenarioId': task.config.scenario_id,
            'replicate': task.replicate,
            'seed': task.config.seed,
            'method': 'fracfixr',
            'methodError': fraction_error,
            'truthHash': truth_hash,
            'simulationHash': simulation_hash,
            'configHash': config_hash,
            **evaluate_fraction_weights(
                fraction_estimates,
                experiment.fraction_truth,
            ),
        }], columns = FRACTION_METRIC_COLUMNS)
    return TaskResult(
        gene_metrics = pd.DataFrame.from_records(records),
        fraction_metrics = fraction_record,
    )


def _metric_record(
    result: MethodResult,
    *,
    experiment: SimulatedExperiment,
    task: BenchmarkTask,
    alpha: float,
    minimum_mean: float,
    truth_hash: str,
    simulation_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    metrics = evaluate_results(
        result.table,
        experiment.truth,
        truth_column = result.truth_column,
        alpha = alpha,
        minimum_mean = minimum_mean,
    )
    return {
        'scenarioId': task.config.scenario_id,
        'replicate': task.replicate,
        'seed': task.config.seed,
        'participantsPerCondition': task.config.participants_per_condition,
        'genes': task.config.genes,
        'redistributedGenes': task.config.redistributed_genes,
        'transcriptionGenes': task.config.transcription_genes,
        'method': result.method,
        'estimand': result.estimand,
        'truthColumn': result.truth_column,
        'normalization': _normalization(result.method),
        'runtimeSeconds': result.runtime_seconds,
        'methodError': result.error,
        'resultHash': _frame_hash(result.table),
        'truthHash': truth_hash,
        'simulationHash': simulation_hash,
        'configHash': config_hash,
        **metrics.to_record(),
    }


def _read_specification(path: Path) -> dict[str, Any]:
    try:
        specification = json.loads(path.read_text(encoding = 'utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Could not read benchmark configuration: {error}') from error
    required = {
        'benchmarkId',
        'alpha',
        'minimumMean',
        'methods',
        'scenarios',
    }
    missing = required.difference(specification)
    if missing:
        raise ValueError(f'Benchmark configuration is missing: {sorted(missing)}.')
    unknown = set(specification).difference(required)
    if unknown:
        raise ValueError(f'Unknown benchmark fields: {sorted(unknown)}.')
    if (
        not isinstance(specification['benchmarkId'], str)
        or not specification['benchmarkId'].strip()
    ):
        raise ValueError('benchmarkId must be a nonempty string.')
    if not _probability(specification['alpha']):
        raise ValueError('alpha must be between zero and one.')
    if not _finite_nonnegative(specification['minimumMean']):
        raise ValueError('minimumMean must be finite and nonnegative.')
    methods = specification['methods']
    if (
        not isinstance(methods, list)
        or not methods
        or len(methods) != len(set(methods))
        or set(methods).difference(METHODS)
    ):
        raise ValueError(f'methods must be unique members of {list(METHODS)}.')
    if (
        not isinstance(specification['scenarios'], list)
        or not specification['scenarios']
    ):
        raise ValueError('scenarios must be a nonempty list.')
    return specification


def _benchmark_tasks(specification: dict[str, Any]) -> list[BenchmarkTask]:
    tasks = []
    scenario_ids = set()
    used_seeds = set()
    for scenario in specification['scenarios']:
        if not isinstance(scenario, dict):
            raise ValueError('Each scenario must be an object.')
        record = dict(scenario)
        try:
            replicates = record.pop('replicates')
            seed_start = record.pop('seedStart')
        except KeyError as error:
            raise ValueError(
                'Each scenario requires replicates and seedStart.',
            ) from error
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates < 1
        ):
            raise ValueError('replicates must be a positive integer.')
        if (
            isinstance(seed_start, bool)
            or not isinstance(seed_start, int)
            or seed_start < 0
        ):
            raise ValueError('seedStart must be a nonnegative integer.')
        scenario_id = record.get('scenarioId')
        if scenario_id in scenario_ids:
            raise ValueError('scenarioId values must be unique.')
        scenario_ids.add(scenario_id)
        for replicate in range(replicates):
            seed = seed_start + replicate
            if seed in used_seeds:
                raise ValueError('Simulation seeds must be globally unique.')
            used_seeds.add(seed)
            tasks.append(BenchmarkTask(
                replicate = replicate,
                config = SimulationConfig.from_record(record, seed = seed),
            ))
    return tasks


def _seed_manifest(
    tasks: list[BenchmarkTask],
    *,
    config_hash: str,
) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {
            'scenarioId': task.config.scenario_id,
            'replicate': task.replicate,
            'seed': task.config.seed,
            'configHash': config_hash,
            'simulationConfig': json.dumps(
                task.config.to_record(),
                sort_keys = True,
                separators = (',', ':'),
            ),
        }
        for task in tasks
    ])


def _part_paths(directory: Path, task: BenchmarkTask) -> tuple[Path, Path]:
    stem = f'{task.config.scenario_id}__r{task.replicate:04d}'
    return directory / f'{stem}.tsv', directory / f'{stem}.fractions.tsv'


def _write_part(directory: Path, task: BenchmarkTask, result: TaskResult) -> None:
    gene_path, fraction_path = _part_paths(directory, task)
    _atomic_frame_write(result.gene_metrics, gene_path)
    _atomic_frame_write(result.fraction_metrics, fraction_path)


def _collect_parts(
    directory: Path,
    tasks: list[BenchmarkTask],
    *,
    methods: tuple[str, ...],
    config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_parts = []
    fraction_parts = []
    for task in tasks:
        gene_path, fraction_path = _part_paths(directory, task)
        if not gene_path.exists() or not fraction_path.exists():
            raise RuntimeError(f'Missing benchmark part for {task}.')
        gene_parts.append(pd.read_csv(
            gene_path,
            sep = '\t',
            keep_default_na = False,
            na_values = [''],
        ))
        fraction = pd.read_csv(
            fraction_path,
            sep = '\t',
            keep_default_na = False,
            na_values = [''],
        )
        if len(fraction):
            fraction_parts.append(fraction)
    genes = pd.concat(gene_parts, ignore_index = True)
    fractions = (
        pd.concat(fraction_parts, ignore_index = True)
        if fraction_parts
        else pd.DataFrame(columns = FRACTION_METRIC_COLUMNS)
    )
    expected = {
        (
            task.config.scenario_id,
            task.replicate,
            method,
            estimand,
        )
        for task in tasks
        for method in methods
        for estimand in METHOD_ESTIMANDS[method]
    }
    key_columns = ['scenarioId', 'replicate', 'method', 'estimand']
    observed = set(genes.loc[:, key_columns].itertuples(index = False, name = None))
    if genes.duplicated(key_columns).any() or expected != observed:
        raise RuntimeError(
            'Collected benchmark parts do not match the method and seed manifest.',
        )
    if not genes['configHash'].eq(config_hash).all():
        raise RuntimeError('Collected benchmark parts have an invalid config hash.')
    _validate_fraction_parts(
        fractions,
        tasks = tasks,
        methods = methods,
        config_hash = config_hash,
    )
    return genes, fractions


def _validate_fraction_parts(
    fractions: pd.DataFrame,
    *,
    tasks: list[BenchmarkTask],
    methods: tuple[str, ...],
    config_hash: str,
) -> None:
    missing = set(FRACTION_METRIC_COLUMNS).difference(fractions.columns)
    if missing:
        raise RuntimeError(f'Fraction parts are missing columns: {sorted(missing)}.')
    expected = (
        {(task.config.scenario_id, task.replicate, 'fracfixr') for task in tasks}
        if 'fracfixr' in methods
        else set()
    )
    key_columns = ['scenarioId', 'replicate', 'method']
    observed = set(
        fractions.loc[:, key_columns].itertuples(index = False, name = None),
    )
    if fractions.duplicated(key_columns).any() or expected != observed:
        raise RuntimeError(
            'Collected fraction parts do not match the method and seed manifest.',
        )
    if len(fractions) and not fractions['configHash'].eq(config_hash).all():
        raise RuntimeError('Collected fraction parts have an invalid config hash.')


def _atomic_frame_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    frame.to_csv(temporary, sep = '\t', index = False)
    temporary.replace(path)


def _validate_existing_manifest(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        return
    observed = json.loads(path.read_text(encoding = 'utf-8'))
    if observed.get('configHash') != expected['configHash']:
        raise ValueError(
            'Output directory contains a different benchmark configuration.',
        )
    if observed.get('methods') != expected['methods']:
        raise ValueError('Output directory contains a different method set.')
    if observed.get('benchmarkCodeHash') != expected['benchmarkCodeHash']:
        raise ValueError('Output directory contains a different benchmark build.')


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(
        json.dumps(value, indent = 2, sort_keys = True) + '\n',
        encoding = 'utf-8',
    )
    temporary.replace(path)


def _environment_record() -> dict[str, Any]:
    return {
        'monopolyPolysome': version('monopoly-polysome'),
        'pydeseq2': version('pydeseq2'),
        'numpy': version('numpy'),
        'pandas': version('pandas'),
        'scipy': version('scipy'),
        'python': platform.python_version(),
        'platform': platform.platform(),
    }


def _report_progress(
    completed: int,
    *,
    total: int,
    benchmark_id: str,
) -> None:
    percent = 100 * completed / total
    print(
        f'{benchmark_id}: {completed}/{total} replicates complete ({percent:.1f}%).',
        flush = True,
    )


def _report_task_failure(task: BenchmarkTask, error: Exception) -> None:
    print(
        f'{task.config.scenario_id} replicate {task.replicate} failed: '
        f'{type(error).__name__}: {error}',
        flush = True,
    )


def _raise_task_failures(
    failures: list[tuple[BenchmarkTask, Exception]],
    *,
    completed: int,
    total: int,
) -> None:
    examples = '; '.join(
        f'{task.config.scenario_id} replicate {task.replicate}: '
        f'{type(error).__name__}: {error}'
        for task, error in failures[:5]
    )
    remaining = len(failures) - 5
    if remaining > 0:
        examples = f'{examples}; and {remaining} more'
    raise RuntimeError(
        f'{len(failures)} benchmark replicate(s) failed; '
        f'{completed}/{total} completed parts were retained. {examples}',
    ) from failures[0][1]


def _r_version_record(*, rscript: str | Path) -> dict[str, str]:
    observed = r_versions(rscript = rscript)
    return dict(zip(observed['package'], observed['version'], strict = True))


def _validate_r_versions(
    observed: dict[str, str],
    *,
    methods: tuple[str, ...],
) -> None:
    packages = {
        'R',
        *(
            {'deseq2': 'DESeq2', 'edgeR': 'edgeR', 'fracfixr': 'FracFixR'}[
                method
            ]
            for method in methods
        ),
    }
    mismatches = {
        package: {
            'expected': R_REQUIREMENTS[package],
            'observed': observed.get(package, 'NA'),
        }
        for package in packages
        if observed.get(package) != R_REQUIREMENTS[package]
    }
    if mismatches:
        raise RuntimeError(f'R versions do not match the benchmark: {mismatches}.')


def _normalization(method: str) -> str:
    return {
        'monopoly': 'fractionMedianRatio',
        'deseq2': 'fractionMedianRatio',
        'edgeR': 'fractionTMM',
        'ratioWelch': 'none',
        'fracfixr': 'fractionReconstruction',
    }[method]


def _simulation_hash(experiment: SimulatedExperiment) -> str:
    digest = hashlib.sha256()
    for frame in (
        experiment.count_matrix(),
        experiment.truth,
        experiment.fraction_truth,
        experiment.dispersions,
    ):
        digest.update(_frame_hash(frame).encode())
    return f'sha256:{digest.hexdigest()}'


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    digest.update('\0'.join(map(str, frame.columns)).encode())
    digest.update('\0'.join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index = True).values.tobytes())
    return f'sha256:{digest.hexdigest()}'


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys = True,
        separators = (',', ':'),
    ).encode()
    return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


def _benchmark_code_hash() -> str:
    root = Path(__file__).parent
    paths = [
        root / '__init__.py',
        root / 'metrics.py',
        root / 'methods.py',
        root / 'run.py',
        root / 'simulate.py',
        root / 'r' / 'requirements.tsv',
        root / 'r' / 'run_comparators.R',
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return f'sha256:{digest.hexdigest()}'


def _probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
        and 0 < value < 1
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
        and value >= 0
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Run the Monopoly benchmark.')
    parser.add_argument('config', type = Path)
    parser.add_argument('output', type = Path)
    parser.add_argument('--rscript', type = Path)
    parser.add_argument('--workers', type = int, default = 1)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = _parse_args()
    run_benchmark(
        arguments.config,
        arguments.output,
        rscript = arguments.rscript,
        workers = arguments.workers,
    )
