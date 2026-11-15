import inspect
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from monopoly import (
    DataValidationError,
    MonopolyDataset,
    MonopolyModel,
    MonopolyResults,
    __version__,
)


def test_public_signatures_follow_the_decision_contract() -> None:
    dataset_parameters = inspect.signature(MonopolyDataset).parameters
    model_parameters = inspect.signature(MonopolyModel).parameters
    fit_parameters = inspect.signature(MonopolyModel.fit).parameters
    contrast_parameters = inspect.signature(MonopolyModel.contrast).parameters
    write_parameters = inspect.signature(MonopolyResults.write).parameters

    assert 'input' in dataset_parameters
    assert 'total' not in dataset_parameters
    assert 'reference' in model_parameters
    assert 'reference_condition' not in model_parameters
    assert 'test_condition' not in model_parameters
    assert fit_parameters['normalization'].default == 'fraction'
    assert list(contrast_parameters)[:3] == ['self', 'numerator', 'denominator']
    assert 'overwrite' not in write_parameters
    assert __version__ == '0.1.0'


def test_source_and_package_versions_match() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding = 'utf-8'))

    assert project['project']['name'] == 'monopoly-polysome'
    assert project['project']['version'] == __version__
    assert project['project']['license'] == 'MIT'
    assert [author['name'] for author in project['project']['authors']] == [
        'William DeGroat',
        'Anat Kreimer',
    ]


def test_retired_total_fraction_is_not_a_compatibility_alias(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    assert dataset.library_metadata['fraction'].drop_duplicates().tolist() == [
        'input',
        'monosome',
        'polysome',
    ]

    with pytest.raises(DataValidationError, match = 'Unknown fraction'):
        dataset.counts('total')
