from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from monopoly import MonopolyDataset

FRACTIONS = ('input', 'monosome', 'polysome')


@dataclass(frozen = True)
class SimulationConfig:
    """Define one composition-consistent simulation replicate."""

    scenario_id: str
    participants_per_condition: int
    genes: int
    redistributed_genes: int = 0
    transcription_genes: int = 0
    redistribution_log2_effect: float = 1.0
    redistribution_positive_fraction: float = 1.0
    transcription_log2_effect: float = 1.0
    baseline_mean: float = 100.0
    abundance_log_sd: float = 1.0
    participant_log_sd: float = 0.3
    library_log_sd: float = 0.3
    fraction_dispersions: tuple[float, float, float] = (0.1, 0.15, 0.3)
    dispersion_log_sd: float = 0.0
    dispersion_correlation: float = 1.0
    associated_fraction: float = 0.9
    polysome_fraction: float = 0.65
    allocation_logit_sd: float = 0.5
    seed: int = 1

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        *,
        seed: int | None = None,
    ) -> SimulationConfig:
        """Construct a configuration from exported camelCase fields."""
        expected = {
            'scenarioId': 'scenario_id',
            'participantsPerCondition': 'participants_per_condition',
            'genes': 'genes',
            'redistributedGenes': 'redistributed_genes',
            'transcriptionGenes': 'transcription_genes',
            'redistributionLog2Effect': 'redistribution_log2_effect',
            'redistributionPositiveFraction': (
                'redistribution_positive_fraction'
            ),
            'transcriptionLog2Effect': 'transcription_log2_effect',
            'baselineMean': 'baseline_mean',
            'abundanceLogSd': 'abundance_log_sd',
            'participantLogSd': 'participant_log_sd',
            'libraryLogSd': 'library_log_sd',
            'fractionDispersions': 'fraction_dispersions',
            'dispersionLogSd': 'dispersion_log_sd',
            'dispersionCorrelation': 'dispersion_correlation',
            'associatedFraction': 'associated_fraction',
            'polysomeFraction': 'polysome_fraction',
            'allocationLogitSd': 'allocation_logit_sd',
            'seed': 'seed',
        }
        unknown = set(record).difference(expected)
        if unknown:
            raise ValueError(f'Unknown simulation fields: {sorted(unknown)}.')
        values = {expected[name]: value for name, value in record.items()}
        if 'fraction_dispersions' in values:
            values['fraction_dispersions'] = tuple(values['fraction_dispersions'])
        if seed is not None:
            values['seed'] = seed
        return cls(**values)

    def to_record(self) -> dict[str, Any]:
        """Return exported camelCase configuration fields."""
        return {
            'scenarioId': self.scenario_id,
            'participantsPerCondition': self.participants_per_condition,
            'genes': self.genes,
            'redistributedGenes': self.redistributed_genes,
            'transcriptionGenes': self.transcription_genes,
            'redistributionLog2Effect': self.redistribution_log2_effect,
            'redistributionPositiveFraction': (
                self.redistribution_positive_fraction
            ),
            'transcriptionLog2Effect': self.transcription_log2_effect,
            'baselineMean': self.baseline_mean,
            'abundanceLogSd': self.abundance_log_sd,
            'participantLogSd': self.participant_log_sd,
            'libraryLogSd': self.library_log_sd,
            'fractionDispersions': list(self.fraction_dispersions),
            'dispersionLogSd': self.dispersion_log_sd,
            'dispersionCorrelation': self.dispersion_correlation,
            'associatedFraction': self.associated_fraction,
            'polysomeFraction': self.polysome_fraction,
            'allocationLogitSd': self.allocation_logit_sd,
            'seed': self.seed,
        }


@dataclass(frozen = True)
class SimulatedExperiment:
    """Contain counts, truth, and expected values for one replicate."""

    dataset: MonopolyDataset
    truth: pd.DataFrame
    library_metadata: pd.DataFrame
    expected_counts: pd.DataFrame
    fraction_truth: pd.DataFrame
    dispersions: pd.DataFrame

    def count_matrix(self) -> pd.DataFrame:
        """Return a gene-by-library matrix for external methods."""
        matrices = []
        for row in self.library_metadata.itertuples(index = False):
            values = self.dataset.counts(row.fraction).loc[row.participantId]
            matrices.append(values.rename(row.sampleId))
        matrix = pd.concat(matrices, axis = 1)
        matrix.index.name = 'geneId'
        return matrix


def simulate_experiment(config: SimulationConfig) -> SimulatedExperiment:
    """Simulate matched input, monosome, and polysome counts."""
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    n = config.participants_per_condition
    participants = pd.Index(
        [f'control{index}' for index in range(n)]
        + [f'test{index}' for index in range(n)],
        name = 'participantId',
    )
    conditions = np.repeat(['Control', 'Test'], n)
    genes = pd.Index([f'gene{index:05d}' for index in range(config.genes)])

    order = rng.permutation(config.genes)
    transcription_index = order[: config.transcription_genes]
    start = config.transcription_genes
    stop = start + config.redistributed_genes
    redistribution_index = order[start:stop]

    transcription = np.zeros(config.genes)
    transcription[transcription_index] = config.transcription_log2_effect
    redistribution = np.zeros(config.genes)
    positive = int(
        np.ceil(
            config.redistributed_genes
            * config.redistribution_positive_fraction,
        ),
    )
    redistribution[redistribution_index[:positive]] = (
        config.redistribution_log2_effect
    )
    redistribution[redistribution_index[positive:]] = (
        -config.redistribution_log2_effect
    )

    baseline = rng.lognormal(
        mean = np.log(config.baseline_mean)
        - 0.5 * config.abundance_log_sd**2,
        sigma = config.abundance_log_sd,
        size = config.genes,
    )
    participant_multiplier = rng.lognormal(
        mean = -0.5 * config.participant_log_sd**2,
        sigma = config.participant_log_sd,
        size = (len(participants), config.genes),
    )
    abundance = baseline[None, :] * participant_multiplier
    abundance *= np.exp2(
        (conditions == 'Test')[:, None] * transcription[None, :],
    )

    allocation_logit = _logit(config.polysome_fraction) + rng.normal(
        0,
        config.allocation_logit_sd,
        size = config.genes,
    )
    control_polysome_share = _expit(allocation_logit)
    test_polysome_share = _expit(
        allocation_logit + np.log(2) * redistribution,
    )
    polysome_share = np.where(
        (conditions == 'Test')[:, None],
        test_polysome_share[None, :],
        control_polysome_share[None, :],
    )
    polysome_allocation = config.associated_fraction * polysome_share
    monosome_allocation = config.associated_fraction * (1 - polysome_share)
    lost_allocation = np.full_like(
        polysome_allocation,
        1 - config.associated_fraction,
    )

    library_factors = rng.lognormal(
        mean = -0.5 * config.library_log_sd**2,
        sigma = config.library_log_sd,
        size = (len(participants), len(FRACTIONS)),
    )
    dispersion_values = _gene_dispersions(rng, config = config)
    allocations = {
        'input': np.ones_like(abundance),
        'monosome': monosome_allocation,
        'polysome': polysome_allocation,
    }
    matrices = {}
    expected_parts = []
    metadata_records = []
    for fraction_index, fraction in enumerate(FRACTIONS):
        mean = (
            abundance
            * allocations[fraction]
            * library_factors[:, fraction_index, None]
        )
        counts = _negative_binomial(
            rng,
            mean = mean,
            dispersion = dispersion_values[:, fraction_index],
        )
        matrices[fraction] = pd.DataFrame(
            counts,
            index = participants,
            columns = genes,
        )
        for participant_index, participant in enumerate(participants):
            sample_id = f'{participant}.{fraction}'
            expected_parts.append(
                pd.Series(mean[participant_index], name = sample_id),
            )
            metadata_records.append({
                'sampleId': sample_id,
                'participantId': participant,
                'condition': conditions[participant_index],
                'fraction': fraction,
            })

    metadata = pd.DataFrame({'condition': conditions}, index = participants)
    library_metadata = pd.DataFrame.from_records(metadata_records)
    expected_counts = pd.concat(expected_parts, axis = 1)
    expected_counts.index = genes
    expected_counts.index.name = 'geneId'

    fraction_truth = _fraction_truth(
        abundance,
        monosome_allocation = monosome_allocation,
        polysome_allocation = polysome_allocation,
        lost_allocation = lost_allocation,
        participants = participants,
        conditions = conditions,
    )
    truth = _gene_truth(
        config,
        genes = genes,
        baseline = baseline,
        transcription = transcription,
        redistribution = redistribution,
        control_polysome_share = control_polysome_share,
        test_polysome_share = test_polysome_share,
    )
    dispersions = pd.DataFrame(
        dispersion_values,
        index = genes,
        columns = FRACTIONS,
    )
    dispersions.index.name = 'geneId'
    dispersions.columns.name = 'fraction'
    return SimulatedExperiment(
        dataset = MonopolyDataset(**matrices, metadata = metadata),
        truth = truth,
        library_metadata = library_metadata,
        expected_counts = expected_counts,
        fraction_truth = fraction_truth,
        dispersions = dispersions,
    )


def _gene_truth(
    config: SimulationConfig,
    *,
    genes: pd.Index,
    baseline: np.ndarray,
    transcription: np.ndarray,
    redistribution: np.ndarray,
    control_polysome_share: np.ndarray,
    test_polysome_share: np.ndarray,
) -> pd.DataFrame:
    control_polysome = config.associated_fraction * control_polysome_share
    test_polysome = config.associated_fraction * test_polysome_share
    control_monosome = config.associated_fraction * (1 - control_polysome_share)
    test_monosome = config.associated_fraction * (1 - test_polysome_share)
    return pd.DataFrame({
        'geneId': genes,
        'meanInputCount': baseline,
        'isRedistributed': redistribution != 0,
        'isTranscriptional': transcription != 0,
        'polyVsMonoLog2FoldChange': redistribution,
        'inputLog2FoldChange': transcription,
        'polysomeProportionLog2FoldChange': np.log2(
            test_polysome / control_polysome,
        ),
        'monosomeProportionLog2FoldChange': np.log2(
            test_monosome / control_monosome,
        ),
        'polysomeOddsLog2FoldChange': (
            _logit(test_polysome) - _logit(control_polysome)
        ) / np.log(2),
        'monosomeOddsLog2FoldChange': (
            _logit(test_monosome) - _logit(control_monosome)
        ) / np.log(2),
    })


def _fraction_truth(
    abundance: np.ndarray,
    *,
    monosome_allocation: np.ndarray,
    polysome_allocation: np.ndarray,
    lost_allocation: np.ndarray,
    participants: pd.Index,
    conditions: np.ndarray,
) -> pd.DataFrame:
    records = []
    allocations = {
        'monosome': monosome_allocation,
        'polysome': polysome_allocation,
        'lost': lost_allocation,
    }
    total = abundance.sum(axis = 1)
    for participant_index, participant in enumerate(participants):
        for fraction, allocation in allocations.items():
            value = (
                abundance[participant_index] * allocation[participant_index]
            ).sum() / total[participant_index]
            records.append({
                'participantId': participant,
                'condition': conditions[participant_index],
                'fraction': fraction,
                'trueFraction': value,
            })
    return pd.DataFrame.from_records(records)


def _negative_binomial(
    rng: np.random.Generator,
    *,
    mean: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    size = 1.0 / dispersion[None, :]
    probability = size / (size + mean)
    return rng.negative_binomial(size, probability).astype(np.int64)


def _gene_dispersions(
    rng: np.random.Generator,
    *,
    config: SimulationConfig,
) -> np.ndarray:
    baseline = np.broadcast_to(
        config.fraction_dispersions,
        (config.genes, len(FRACTIONS)),
    ).astype(np.float64, copy = True)
    if config.dispersion_log_sd == 0:
        return baseline
    common = rng.normal(size = (config.genes, 1))
    independent = rng.normal(size = baseline.shape)
    multiplier = np.exp(
        config.dispersion_log_sd
        * (
            np.sqrt(config.dispersion_correlation) * common
            + np.sqrt(1 - config.dispersion_correlation) * independent
        )
        - 0.5 * config.dispersion_log_sd**2,
    )
    return baseline * multiplier


def _validate_config(config: SimulationConfig) -> None:
    if not isinstance(config.scenario_id, str) or not config.scenario_id.strip():
        raise ValueError('scenario_id must be a nonempty string.')
    _validate_config_counts(config)
    _validate_config_numbers(config)
    _validate_config_ranges(config)
    if not _is_integer(config.seed, minimum = 0):
        raise ValueError('seed must be a nonnegative integer.')


def _validate_config_counts(config: SimulationConfig) -> None:
    for name, minimum in (
        ('participants_per_condition', 2),
        ('genes', 20),
        ('redistributed_genes', 0),
        ('transcription_genes', 0),
    ):
        if not _is_integer(getattr(config, name), minimum = minimum):
            raise ValueError(f'{name} must be an integer of at least {minimum}.')
    if config.redistributed_genes + config.transcription_genes > config.genes:
        raise ValueError('Effect gene counts cannot exceed genes.')


def _validate_config_numbers(config: SimulationConfig) -> None:
    for name in (
        'redistribution_log2_effect',
        'redistribution_positive_fraction',
        'transcription_log2_effect',
        'baseline_mean',
        'abundance_log_sd',
        'participant_log_sd',
        'library_log_sd',
        'dispersion_log_sd',
        'dispersion_correlation',
        'associated_fraction',
        'polysome_fraction',
        'allocation_logit_sd',
    ):
        if not _is_finite_real(getattr(config, name)):
            raise ValueError(f'{name} must be finite.')


def _validate_config_ranges(config: SimulationConfig) -> None:
    if config.baseline_mean <= 0:
        raise ValueError('baseline_mean must be positive.')
    if any(
        getattr(config, name) < 0
        for name in (
            'abundance_log_sd',
            'participant_log_sd',
            'library_log_sd',
            'dispersion_log_sd',
            'allocation_logit_sd',
        )
    ):
        raise ValueError('Standard deviations cannot be negative.')
    if not 0 <= config.dispersion_correlation <= 1:
        raise ValueError('dispersion_correlation must be between zero and one.')
    if not 0 <= config.redistribution_positive_fraction <= 1:
        raise ValueError(
            'redistribution_positive_fraction must be between zero and one.',
        )
    if not 0 < config.associated_fraction <= 1:
        raise ValueError('associated_fraction must be in (0, 1].')
    if not 0 < config.polysome_fraction < 1:
        raise ValueError('polysome_fraction must be between zero and one.')
    if (
        not isinstance(config.fraction_dispersions, tuple)
        or len(config.fraction_dispersions) != len(FRACTIONS)
        or any(
            not _is_finite_real(value) or value <= 0
            for value in config.fraction_dispersions
        )
    ):
        raise ValueError('fraction_dispersions must contain three positive values.')


def _logit(values: float | np.ndarray) -> float | np.ndarray:
    return np.log(values / (1 - values))


def _expit(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def _is_integer(value: object, *, minimum: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and value >= minimum
    )


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and bool(np.isfinite(value))
    )
