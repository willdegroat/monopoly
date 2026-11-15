import numpy as np
import pandas as pd
import pytest

from monopoly import DataValidationError, MonopolyResults


@pytest.fixture
def results() -> MonopolyResults:
    return MonopolyResults(
        table = pd.DataFrame({'geneId': ['gene1'], 'pValue': [0.1]}),
        diagnostics = pd.DataFrame({'geneId': ['gene1']}),
        provenance = {
            'contrast': {'numerator': 'polysome', 'denominator': 'monosome'},
        },
    )


def test_results_repr_is_concise(results: MonopolyResults) -> None:
    assert repr(results) == (
        "MonopolyResults(contrast = 'polysome' vs 'monosome', genes = 1)"
    )


@pytest.mark.parametrize(
    ('path', 'message'),
    [
        ('results.txt', r'\.csv or \.tsv'),
        (None, 'string or pathlib.Path'),
    ],
)
def test_results_reject_invalid_paths(
    results: MonopolyResults,
    path: object,
    message: str,
) -> None:
    with pytest.raises(DataValidationError, match = message):
        results.write(path)  # type: ignore[arg-type]


def test_results_reject_missing_output_directory(
    results: MonopolyResults,
    tmp_path,
) -> None:
    with pytest.raises(DataValidationError, match = 'does not exist'):
        results.write(tmp_path / 'missing' / 'results.tsv')


def test_results_reject_file_as_output_parent(
    results: MonopolyResults,
    tmp_path,
) -> None:
    parent = tmp_path / 'file'
    parent.write_text('content', encoding = 'utf-8')

    with pytest.raises(DataValidationError, match = 'not a directory'):
        results.write(parent / 'results.tsv')


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ('tableType', 'table must be a pandas DataFrame'),
        ('diagnosticsType', 'diagnostics must be a pandas DataFrame'),
        ('missingGeneId', 'contain a geneId'),
        ('duplicateGeneId', 'must be unique'),
        ('emptyGeneId', 'nonempty strings'),
        ('misaligned', 'identical order'),
        ('provenanceType', 'provenance must be a dictionary'),
        ('provenanceValue', 'JSON-serializable'),
        ('provenanceNonfinite', 'finite'),
    ],
)
def test_results_reject_invalid_constructor_inputs(
    mutation: str,
    message: str,
) -> None:
    table: object = pd.DataFrame({'geneId': ['gene1', 'gene2']})
    diagnostics: object = pd.DataFrame({'geneId': ['gene1', 'gene2']})
    provenance: object = {}
    if mutation == 'tableType':
        table = []
    elif mutation == 'diagnosticsType':
        diagnostics = []
    elif mutation == 'missingGeneId':
        table = pd.DataFrame({'pValue': [0.1, 0.2]})
    elif mutation == 'duplicateGeneId':
        table.loc[1, 'geneId'] = 'gene1'
    elif mutation == 'emptyGeneId':
        table.loc[1, 'geneId'] = ''
    elif mutation == 'misaligned':
        diagnostics = diagnostics.iloc[::-1].reset_index(drop = True)
    elif mutation == 'provenanceType':
        provenance = []
    elif mutation == 'provenanceNonfinite':
        provenance = {'invalid': np.nan}
    else:
        provenance = {'invalid': {1, 2}}

    with pytest.raises(DataValidationError, match = message):
        MonopolyResults(
            table = table,  # type: ignore[arg-type]
            diagnostics = diagnostics,  # type: ignore[arg-type]
            provenance = provenance,  # type: ignore[arg-type]
        )
