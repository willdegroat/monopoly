import pandas as pd
import pytest


@pytest.fixture
def dataset_inputs() -> dict[str, pd.DataFrame]:
    participants = ['controlA', 'controlB', 'testA', 'testB']
    genes = ['geneA', 'geneB', 'geneZero']

    return {
        'input': pd.DataFrame(
            [[10, 20, 0], [30, 40, 0], [50, 60, 0], [70, 80, 0]],
            index = participants,
            columns = genes,
        ),
        'monosome': pd.DataFrame(
            [[4, 8, 0], [12, 16, 0], [20, 24, 0], [28, 32, 0]],
            index = participants,
            columns = genes,
        ),
        'polysome': pd.DataFrame(
            [[6, 12, 0], [18, 24, 0], [30, 36, 0], [42, 48, 0]],
            index = participants,
            columns = genes,
        ),
        'metadata': pd.DataFrame(
            {
                'condition': ['Control', 'Control', 'Test', 'Test'],
                'age': [20, 22, 21, 23],
            },
            index = participants,
        ),
    }


@pytest.fixture
def library_metadata(dataset_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    libraries = (
        (participant, fraction)
        for participant in dataset_inputs['metadata'].index
        for fraction in ('input', 'monosome', 'polysome')
    )
    rows = [
        {
            'participantId': participant,
            'fraction': fraction,
            'batch': f'batch{position % 2 + 1}',
        }
        for position, (participant, fraction) in enumerate(libraries)
    ]
    return pd.DataFrame(rows)
