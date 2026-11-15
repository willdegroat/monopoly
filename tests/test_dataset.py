import numpy as np
import pandas as pd
import pytest

from monopoly import DataValidationError, MonopolyDataset


def test_dataset_aligns_labels_and_copies_inputs(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['monosome'] = dataset_inputs['monosome'].loc[
        ['testB', 'testA', 'controlB', 'controlA'],
        ['geneZero', 'geneB', 'geneA'],
    ]
    dataset = MonopolyDataset(**dataset_inputs)
    dataset_inputs['input'].iloc[0, 0] = 999
    dataset_inputs['metadata'].iloc[0, 0] = 'Changed'

    assert dataset.participants.tolist() == [
        'controlA',
        'controlB',
        'testA',
        'testB',
    ]
    assert dataset.genes.tolist() == ['geneA', 'geneB', 'geneZero']
    assert dataset.counts('input').iloc[0, 0] == 10
    assert dataset.counts('monosome').iloc[0, 0] == 4
    assert dataset.metadata.iloc[0]['condition'] == 'Control'
    assert repr(dataset) == 'MonopolyDataset(participants = 4, genes = 3)'


def test_dataset_accepts_whole_floats(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['polysome'] = dataset_inputs['polysome'].astype(float)

    dataset = MonopolyDataset(**dataset_inputs)

    assert all(
        dtype == np.dtype('int64')
        for dtype in dataset.counts('polysome').dtypes
    )


def test_dataset_rejects_fractional_estimated_counts_with_guidance(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['input'] = dataset_inputs['input'].astype(float)
    dataset_inputs['input'].iloc[0, 0] = 10.001

    with pytest.raises(DataValidationError, match = 'Round estimated gene counts'):
        MonopolyDataset(**dataset_inputs)


def test_dataset_outputs_are_defensive_copies(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)
    counts = dataset.counts('input')
    metadata = dataset.metadata
    library_metadata = dataset.library_metadata

    counts.iloc[0, 0] = 999
    metadata.iloc[0, 0] = 'Changed'
    library_metadata.iloc[0, 0] = 'Changed'

    assert dataset.counts('input').iloc[0, 0] == 10
    assert dataset.metadata.iloc[0, 0] == 'Control'
    assert dataset.library_metadata.iloc[0, 0] == 'controlA'
    assert dataset.gene_metadata is None


@pytest.mark.parametrize(
    ('value', 'message'),
    [
        (1.5, 'whole numbers'),
        (-1, 'negative'),
        (np.nan, 'NA'),
        (np.inf, 'finite'),
    ],
)
def test_dataset_rejects_invalid_numeric_counts(
    dataset_inputs: dict[str, pd.DataFrame],
    value: float,
    message: str,
) -> None:
    dataset_inputs['polysome'] = dataset_inputs['polysome'].astype(float)
    dataset_inputs['polysome'].iloc[0, 0] = value

    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize('value', [True, '1', 1 + 2j])
def test_dataset_rejects_non_count_dtypes(
    dataset_inputs: dict[str, pd.DataFrame],
    value: object,
) -> None:
    dataset_inputs['input']['geneA'] = value

    with pytest.raises(DataValidationError):
        MonopolyDataset(**dataset_inputs)


def test_dataset_rejects_integer_overflow(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['input']['geneA'] = np.array(
        [np.iinfo(np.uint64).max] * 4,
        dtype = np.uint64,
    )

    with pytest.raises(DataValidationError, match = 'integer range'):
        MonopolyDataset(**dataset_inputs)


def test_dataset_rejects_float_overflow(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['input'] = dataset_inputs['input'].astype(float)
    dataset_inputs['input'].iloc[0, 0] = 1e20

    with pytest.raises(DataValidationError, match = 'integer range'):
        MonopolyDataset(**dataset_inputs)


def test_dataset_rejects_negative_integer_counts(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['input'].iloc[0, 0] = -1

    with pytest.raises(DataValidationError, match = 'negative'):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize('replacement', [[], pd.DataFrame()])
def test_dataset_rejects_invalid_count_frames(
    dataset_inputs: dict[str, pd.DataFrame],
    replacement: object,
) -> None:
    dataset_inputs['input'] = replacement  # type: ignore[assignment]

    with pytest.raises(DataValidationError):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_dataset_rejects_invalid_count_labels(
    dataset_inputs: dict[str, pd.DataFrame],
    axis: str,
) -> None:
    if axis == 'index':
        dataset_inputs['input'].index = ['', 'controlB', 'testA', 'testB']
    else:
        dataset_inputs['input'].columns = ['geneA', 'geneA', 'geneZero']

    with pytest.raises(DataValidationError, match = 'identifiers'):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize('axis', ['participants', 'genes'])
def test_dataset_rejects_mismatched_count_labels(
    dataset_inputs: dict[str, pd.DataFrame],
    axis: str,
) -> None:
    if axis == 'participants':
        dataset_inputs['monosome'] = dataset_inputs['monosome'].rename(
            index = {'testB': 'other'},
        )
    else:
        dataset_inputs['monosome'] = dataset_inputs['monosome'].rename(
            columns = {'geneB': 'other'},
        )

    with pytest.raises(DataValidationError, match = f'{axis} do not match'):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ('missingCondition', 'condition'),
        ('missingValue', 'cannot be missing'),
        ('oneLevel', 'exactly two'),
        ('oneParticipant', 'at least two'),
        ('reservedColumn', 'reserved columns'),
    ],
)
def test_dataset_rejects_invalid_participant_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
    mutation: str,
    message: str,
) -> None:
    metadata = dataset_inputs['metadata']
    if mutation == 'missingCondition':
        dataset_inputs['metadata'] = metadata.drop(columns = 'condition')
    elif mutation == 'missingValue':
        metadata.loc['controlA', 'condition'] = None
    elif mutation == 'oneLevel':
        metadata['condition'] = 'Control'
    elif mutation == 'oneParticipant':
        metadata.loc['controlB', 'condition'] = 'Test'
    else:
        metadata['participantId'] = metadata.index

    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(**dataset_inputs)


@pytest.mark.parametrize(
    ('replacement', 'message'),
    [
        ([], 'pandas DataFrame'),
        (42, 'nonempty strings'),
        ('', 'nonempty strings'),
    ],
)
def test_dataset_rejects_invalid_metadata_types(
    dataset_inputs: dict[str, pd.DataFrame],
    replacement: object,
    message: str,
) -> None:
    if isinstance(replacement, list):
        dataset_inputs['metadata'] = replacement  # type: ignore[assignment]
    else:
        dataset_inputs['metadata']['condition'] = dataset_inputs['metadata'][
            'condition'
        ].astype(object)
        dataset_inputs['metadata'].loc['controlA', 'condition'] = replacement

    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(**dataset_inputs)


def test_dataset_rejects_mismatched_metadata_participants(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset_inputs['metadata'] = dataset_inputs['metadata'].rename(
        index = {'testB': 'other'},
    )

    with pytest.raises(DataValidationError, match = 'participants do not match'):
        MonopolyDataset(**dataset_inputs)


def test_dataset_aligns_gene_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    gene_metadata = pd.DataFrame(
        {'geneName': ['ZERO', 'B', 'A']},
        index = ['geneZero', 'geneB', 'geneA'],
    )

    dataset = MonopolyDataset(**dataset_inputs, gene_metadata = gene_metadata)
    gene_metadata.iloc[0, 0] = 'Changed'

    assert dataset.gene_metadata is not None
    assert dataset.gene_metadata['geneName'].tolist() == ['A', 'B', 'ZERO']


@pytest.mark.parametrize(
    ('gene_metadata', 'message'),
    [
        ([], 'pandas DataFrame'),
        (
            pd.DataFrame(
                {'geneName': ['A', 'B', 'C']},
                index = ['geneA', 'geneB', 'other'],
            ),
            'genes do not match',
        ),
        (
            pd.DataFrame(
                {'geneId': ['geneA', 'geneB', 'geneZero']},
                index = ['geneA', 'geneB', 'geneZero'],
            ),
            'reserved',
        ),
    ],
)
def test_dataset_rejects_invalid_gene_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
    gene_metadata: object,
    message: str,
) -> None:
    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(
            **dataset_inputs,
            gene_metadata = gene_metadata,  # type: ignore[arg-type]
        )


def test_dataset_constructs_default_library_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    assert dataset.library_metadata.iloc[:4].to_dict('records') == [
        {'participantId': 'controlA', 'fraction': 'input'},
        {'participantId': 'controlA', 'fraction': 'monosome'},
        {'participantId': 'controlA', 'fraction': 'polysome'},
        {'participantId': 'controlB', 'fraction': 'input'},
    ]


def test_dataset_aligns_supplied_library_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
) -> None:
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata.sample(frac = 1, random_state = 7),
    )
    library_metadata.iloc[0, 2] = 'Changed'

    assert dataset.library_metadata.iloc[0].to_dict() == {
        'participantId': 'controlA',
        'fraction': 'input',
        'batch': 'batch1',
    }


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ('missingColumn', 'missing required columns'),
        ('duplicate', 'duplicate participant-fraction'),
        ('missingRow', 'rows do not match'),
        ('unexpectedFraction', 'rows do not match'),
        ('overlap', 'columns overlap'),
    ],
)
def test_dataset_rejects_invalid_library_metadata(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
    mutation: str,
    message: str,
) -> None:
    if mutation == 'missingColumn':
        library_metadata = library_metadata.drop(columns = 'fraction')
    elif mutation == 'duplicate':
        library_metadata.iloc[1, :2] = library_metadata.iloc[0, :2]
    elif mutation == 'missingRow':
        library_metadata = library_metadata.iloc[:-1]
    elif mutation == 'unexpectedFraction':
        library_metadata.loc[0, 'fraction'] = 'ribosome'
    else:
        library_metadata['age'] = 1

    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(
            **dataset_inputs,
            library_metadata = library_metadata,
        )


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ('wrongType', 'pandas DataFrame'),
        ('missingValue', 'cannot be missing'),
        ('emptyValue', 'nonempty strings'),
        ('duplicateColumn', 'identifiers must be unique'),
    ],
)
def test_dataset_rejects_invalid_library_metadata_values(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
    mutation: str,
    message: str,
) -> None:
    supplied: object = library_metadata
    if mutation == 'wrongType':
        supplied = []
    elif mutation == 'missingValue':
        library_metadata.loc[0, 'fraction'] = None
    elif mutation == 'emptyValue':
        library_metadata.loc[0, 'participantId'] = ''
    else:
        library_metadata.columns = ['participantId', 'fraction', 'fraction']

    with pytest.raises(DataValidationError, match = message):
        MonopolyDataset(
            **dataset_inputs,
            library_metadata = supplied,  # type: ignore[arg-type]
        )


def test_dataset_long_data_are_aligned(
    dataset_inputs: dict[str, pd.DataFrame],
    library_metadata: pd.DataFrame,
) -> None:
    dataset = MonopolyDataset(
        **dataset_inputs,
        library_metadata = library_metadata,
    )

    counts, metadata = dataset._to_long()

    assert counts.index.equals(metadata.index)
    assert counts.index.names == ['participantId', 'fraction']
    assert counts.loc[('controlA', 'monosome'), 'geneA'] == 4
    assert metadata.loc[('testA', 'polysome'), 'condition'] == 'Test'
    assert metadata.loc[('testA', 'polysome'), 'batch'] == 'batch1'


def test_dataset_rejects_unknown_fraction(
    dataset_inputs: dict[str, pd.DataFrame],
) -> None:
    dataset = MonopolyDataset(**dataset_inputs)

    with pytest.raises(DataValidationError, match = 'Unknown fraction'):
        dataset.counts('poly')
