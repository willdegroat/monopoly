# Monopoly

Monopoly is a Python package for paired differential analysis of input, monosome,
and polysome RNA-seq count data. It estimates condition-associated changes in RNA
distribution between experimental fractions while accounting for the matched
participant design.

Monopoly is designed for experiments in which every participant contributes all
three fractions. It supports larger two-condition studies with unequal condition
sizes; no study-specific sample size is hardcoded.

## Install

```console
python -m pip install monopoly-polysome
```

The distribution is named `monopoly-polysome`; the Python import is `monopoly`.
Python 3.11 or later is required.

## Input

Provide one participant-by-gene count matrix for each fraction and a participant
metadata table containing a lowercase `condition` column. The three matrices must
contain identical participant and gene identifiers.

Counts must be finite, nonnegative whole numbers. For Salmon gene quantification,
round `lengthScaledTPM` count-scale values once before constructing the dataset. Do
not use TPM as counts. Filtering and transcript aggregation remain upstream analysis
decisions; see the
[data contract](https://github.com/WillDeGroat/monopoly/blob/main/docs/DATA_CONTRACT.md).

## Workflow

```python
from monopoly import MonopolyDataset, MonopolyModel

dataset = MonopolyDataset(
    input = input_counts,
    monosome = monosome_counts,
    polysome = polysome_counts,
    metadata = metadata,
    gene_metadata = gene_metadata,
)

model = MonopolyModel(
    dataset = dataset,
    reference = 'Control',
).fit(
    normalization = 'fraction',
)

results = model.contrast('poly', 'mono')
results.write('polyVsMono.tsv')
```

`poly` and `mono` are accepted contrast aliases for `polysome` and `monosome`.
`input` is used without an alias. Version 0.1 requires exactly two conditions and
infers the test condition as the level other than `reference`.

A positive polysome-versus-monosome `log2FoldChange` means that the normalized
polysome-to-monosome ratio is higher in the test condition than in the reference
condition. `poly` versus `input` and `mono` versus `input` provide secondary
decompositions.

## Model

Monopoly fits a negative-binomial generalized linear model with participant fixed
effects, fraction effects, and condition-by-fraction interactions. Participant fixed
effects make each contrast a within-participant fraction comparison. The released
model uses one moderated dispersion per gene across fractions and asymptotic Wald
tests through PyDESeq2.

Fraction-wise median-of-ratios normalization is the default. It avoids treating
input, monosome, and polysome libraries as one compositionally homogeneous source.
Joint normalization remains available as an explicit sensitivity analysis:

```python
model.fit(normalization = 'joint')
```

The complete mathematical specification and assumptions are in
[method specification](https://github.com/WillDeGroat/monopoly/blob/main/docs/METHODS.md).

## Interpretation boundary

Monopoly estimates relative redistribution among sequenced RNA pools. It does not
measure absolute protein synthesis, absolute RNA allocation, or ribosome number per
transcript. Monosome fractions may contain actively translated RNA and must not be
described as untranslated RNA.

The Wald test assumes that one moderated dispersion per gene adequately represents
all modeled fractions. Strong fraction-specific mean-variance differences can
invalidate that assumption. Small studies can still use Monopoly for effect
estimation, ranking, diagnostics, and sensitivity analyses, but the package does not
claim that gene-level false-discovery rates are calibrated merely because a model
can be fit. Larger studies should also inspect fraction-specific variability and
perform design-appropriate sensitivity analyses.

Participant fixed effects leave each modeled library cell unique, so automatic
Cook's filtering and outlier replacement are disabled. Cook's distances and explicit
fit statuses remain available as diagnostics.

## Results

`MonopolyResults.table` contains stable camelCase columns:

- `geneId` and optional `geneName`;
- `baseMean`, `log2FoldChange`, and `log2FoldChangeSE`;
- `statistic`, `pValue`, and `adjustedPValue`.

`MonopolyResults.diagnostics` reports fit status, dispersion, maximum Cook's
distance, Cook's-outlier status, and independent-filter status for every gene.
`MonopolyResults.provenance` records package versions, input fingerprints, design,
normalization, fit settings, contrast direction, and inferential scope.

`results.write('polyVsMono.tsv')` writes three files:

- `polyVsMono.tsv`;
- `polyVsMono.diagnostics.tsv`; and
- `polyVsMono.provenance.json`.

Existing files are replaced.

## Development

```console
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest --cov=monopoly
python -m build
```

The release test suite covers validation, design identifiability, normalization,
contrast direction, invariance, diagnostics, output schemas, provenance, and package
construction. See the
[validation scope](https://github.com/WillDeGroat/monopoly/blob/main/docs/VALIDATION.md).

The public simulation benchmark independently compares Monopoly with R DESeq2,
edgeR, a log-ratio test, and FracFixR. Its estimands, frozen configurations, and run
instructions are in the
[benchmark README](https://github.com/WillDeGroat/monopoly/blob/main/benchmarks/README.md);
the [HPC guide](https://github.com/WillDeGroat/monopoly/blob/main/docs/HPC_BENCHMARKS.md)
covers publication-scale execution. FracFixR
results are not presented as the same contrast when its native test targets a
different single-fraction proportion.

Monopoly is developed by William DeGroat and Anat Kreimer and released under the
[MIT License](https://github.com/WillDeGroat/monopoly/blob/main/LICENSE).

## Publication benchmark

The complete frozen study benchmarks and the additional equal-dispersion run
are linked from [benchmark results](benchmarks/results/README.md).
See [release instructions](docs/RELEASING.md) for PyPI Trusted Publishing.
