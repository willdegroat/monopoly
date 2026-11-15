from dataclasses import replace

import pandas as pd
import pytest

from benchmarks.simulate import SimulationConfig, simulate_experiment


@pytest.fixture
def simulation_config() -> SimulationConfig:
    return SimulationConfig(
        scenario_id = 'test',
        participants_per_condition = 3,
        genes = 100,
        redistributed_genes = 20,
        transcription_genes = 15,
        redistribution_log2_effect = 1.5,
        redistribution_positive_fraction = 0.5,
        dispersion_log_sd = 0.4,
        dispersion_correlation = 0.6,
        seed = 1077,
    )


def test_simulation_is_deterministic_and_aligned(
    simulation_config: SimulationConfig,
) -> None:
    first = simulate_experiment(simulation_config)
    second = simulate_experiment(simulation_config)

    pd.testing.assert_frame_equal(first.count_matrix(), second.count_matrix())
    pd.testing.assert_frame_equal(first.truth, second.truth)
    pd.testing.assert_frame_equal(first.fraction_truth, second.fraction_truth)
    assert first.count_matrix().columns.tolist() == first.library_metadata[
        'sampleId'
    ].tolist()
    assert first.count_matrix().shape == (100, 18)
    assert first.truth['geneId'].is_unique


def test_simulation_separates_transcription_and_redistribution(
    simulation_config: SimulationConfig,
) -> None:
    experiment = simulate_experiment(simulation_config)
    truth = experiment.truth

    redistributed = truth['isRedistributed']
    transcriptional = truth['isTranscriptional']
    assert redistributed.sum() == 20
    assert transcriptional.sum() == 15
    assert not (redistributed & transcriptional).any()
    assert set(truth.loc[redistributed, 'polyVsMonoLog2FoldChange']) == {
        -1.5,
        1.5,
    }
    assert truth.loc[transcriptional, 'polyVsMonoLog2FoldChange'].eq(0).all()
    assert truth.loc[redistributed, 'inputLog2FoldChange'].eq(0).all()
    assert truth.loc[~redistributed, 'polysomeProportionLog2FoldChange'].eq(
        0,
    ).all()
    assert truth.loc[~redistributed, 'monosomeProportionLog2FoldChange'].eq(
        0,
    ).all()


def test_fraction_truth_obeys_mass_balance(
    simulation_config: SimulationConfig,
) -> None:
    experiment = simulate_experiment(simulation_config)
    totals = experiment.fraction_truth.groupby('participantId')['trueFraction'].sum()

    assert totals.to_numpy() == pytest.approx(1.0)
    lost = experiment.fraction_truth.query("fraction == 'lost'")['trueFraction']
    assert lost.to_numpy() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('participants_per_condition', 1, 'participants_per_condition'),
        ('genes', 19, 'genes'),
        ('redistributed_genes', 101, 'Effect gene counts'),
        ('redistribution_positive_fraction', 1.1, 'positive_fraction'),
        ('fraction_dispersions', (0.1, 0.2), 'three positive'),
        ('dispersion_correlation', -0.1, 'between zero and one'),
        ('associated_fraction', 0.0, 'associated_fraction'),
        ('seed', -1, 'seed'),
    ],
)
def test_simulation_rejects_invalid_configuration(
    simulation_config: SimulationConfig,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match = message):
        simulate_experiment(replace(simulation_config, **{field: value}))


def test_configuration_round_trip_rejects_unknown_fields(
    simulation_config: SimulationConfig,
) -> None:
    observed = SimulationConfig.from_record(simulation_config.to_record())
    assert observed == simulation_config

    record = simulation_config.to_record()
    record['unexpected'] = 1
    with pytest.raises(ValueError, match = 'Unknown simulation fields'):
        SimulationConfig.from_record(record)
