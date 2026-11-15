import pandas as pd

from monopoly._provenance import _frame_fingerprint


def test_frame_fingerprint_is_stable_and_value_sensitive() -> None:
    frame = pd.DataFrame(
        {'condition': ['Control', 'Test']},
        index = ['sample1', 'sample2'],
    )

    first = _frame_fingerprint(frame)
    second = _frame_fingerprint(frame.copy(deep = True))
    changed = frame.copy(deep = True)
    changed.loc['sample2', 'condition'] = 'Control'

    assert first == second
    assert first != _frame_fingerprint(changed)
    assert first.startswith('sha256:')


def test_frame_fingerprint_handles_unhashable_metadata() -> None:
    frame = pd.DataFrame({'details': [{'batch': 1}, {'batch': 2}]})

    assert _frame_fingerprint(frame) == _frame_fingerprint(frame.copy(deep = True))
