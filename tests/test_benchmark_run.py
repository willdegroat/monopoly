import json
from concurrent.futures import Future
from pathlib import Path

import pandas as pd
import pytest

import benchmarks.run as benchmark_run
from benchmarks.run import (
    FRACTION_METRIC_COLUMNS,
    _benchmark_tasks,
    _read_specification,
    run_benchmark,
)


def _write_config(path: Path, *, benchmark_id: str = 'test') -> None:
    path.write_text(json.dumps({
        'benchmarkId': benchmark_id,
        'alpha': 0.05,
        'minimumMean': 20,
        'methods': ['monopoly', 'ratioWelch'],
        'scenarios': [{
            'scenarioId': 'null',
            'participantsPerCondition': 3,
            'genes': 60,
            'redistributedGenes': 0,
            'transcriptionGenes': 0,
            'replicates': 2,
            'seedStart': 701,
        }],
    }), encoding = 'utf-8')


def test_python_benchmark_runs_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / 'config.json'
    output = tmp_path / 'results'
    _write_config(config)

    run_benchmark(config, output)
    first = json.loads((output / 'manifest.json').read_text(encoding = 'utf-8'))
    run_benchmark(config, output)
    second = json.loads((output / 'manifest.json').read_text(encoding = 'utf-8'))

    assert first['status'] == 'complete'
    assert first['scientificResultHash'] == second['scientificResultHash']
    observed = pd.read_csv(output / 'replicateResults.tsv', sep = '\t')
    assert len(observed) == 4
    assert not observed['methodError'].notna().any()
    fractions = pd.read_csv(output / 'fractionRecovery.tsv', sep = '\t')
    assert fractions.empty
    assert fractions.columns.tolist() == FRACTION_METRIC_COLUMNS


def test_parallel_failure_retains_other_completed_parts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            return future

    config = tmp_path / 'config.json'
    output = tmp_path / 'results'
    _write_config(config)
    original = benchmark_run._run_task

    def fail_first(task, *args):
        if task.replicate == 0:
            raise RuntimeError('simulated worker failure')
        return original(task, *args)

    monkeypatch.setattr(benchmark_run, 'ProcessPoolExecutor', ImmediateExecutor)
    monkeypatch.setattr(benchmark_run, 'as_completed', iter)
    monkeypatch.setattr(benchmark_run, '_run_task', fail_first)

    with pytest.raises(RuntimeError, match = '1/2 completed parts were retained'):
        run_benchmark(config, output, workers = 2)

    parts = output / 'parts'
    assert not (parts / 'null__r0000.tsv').exists()
    assert (parts / 'null__r0001.tsv').exists()
    assert (parts / 'null__r0001.fractions.tsv').exists()


def test_seed_manifest_rejects_duplicate_seeds(tmp_path: Path) -> None:
    config = tmp_path / 'config.json'
    _write_config(config)
    specification = _read_specification(config)
    duplicate = dict(specification['scenarios'][0])
    duplicate['scenarioId'] = 'duplicate'
    duplicate['seedStart'] = 702
    specification['scenarios'].append(duplicate)

    with pytest.raises(ValueError, match = 'globally unique'):
        _benchmark_tasks(specification)


def test_resume_rejects_a_changed_configuration(tmp_path: Path) -> None:
    config = tmp_path / 'config.json'
    output = tmp_path / 'results'
    _write_config(config)
    run_benchmark(config, output)
    _write_config(config, benchmark_id = 'changed')

    with pytest.raises(ValueError, match = 'different benchmark configuration'):
        run_benchmark(config, output)


def test_resume_rejects_a_changed_benchmark_build(tmp_path: Path) -> None:
    config = tmp_path / 'config.json'
    output = tmp_path / 'results'
    _write_config(config)
    run_benchmark(config, output)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding = 'utf-8'))
    manifest['benchmarkCodeHash'] = 'sha256:changed'
    manifest_path.write_text(json.dumps(manifest), encoding = 'utf-8')

    with pytest.raises(ValueError, match = 'different benchmark build'):
        run_benchmark(config, output)


def test_resume_detects_an_incomplete_method_part(tmp_path: Path) -> None:
    config = tmp_path / 'config.json'
    output = tmp_path / 'results'
    _write_config(config)
    run_benchmark(config, output)
    part = output / 'parts' / 'null__r0000.tsv'
    observed = pd.read_csv(part, sep = '\t')
    observed.iloc[:-1].to_csv(part, sep = '\t', index = False)

    with pytest.raises(RuntimeError, match = 'method and seed manifest'):
        run_benchmark(config, output)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('benchmarkId', '', 'nonempty string'),
        ('alpha', True, 'between zero and one'),
        ('alpha', 1.0, 'between zero and one'),
        ('minimumMean', -1, 'finite and nonnegative'),
        ('minimumMean', float('inf'), 'finite and nonnegative'),
    ],
)
def test_specification_rejects_invalid_scalar_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config = tmp_path / 'config.json'
    _write_config(config)
    specification = json.loads(config.read_text(encoding = 'utf-8'))
    specification[field] = value
    config.write_text(json.dumps(specification), encoding = 'utf-8')

    with pytest.raises(ValueError, match = message):
        _read_specification(config)
