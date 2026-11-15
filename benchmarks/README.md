# Monopoly benchmark

This benchmark evaluates the released three-pool model against independently
generated truth. It is not a search for settings that maximize discoveries.

## Methods

- `monopoly`: fraction-wise median-ratio normalization and the released PyDESeq2
  participant-by-fraction interaction model.
- `deseq2`: an independent R implementation of the same design and normalization.
  This is a numerical reference, not a method Monopoly is expected to outperform.
- `edgeR`: the corresponding robust quasi-likelihood interaction model with
  fraction-wise TMM normalization.
- `ratioWelch`: a deliberately simple pseudocount log-ratio followed by a Welch
  test. It represents a common analysis that does not model count uncertainty.
- `fracfixr`: FracFixR 1.1.0 fraction reconstruction followed by its native GLM,
  Logit, and beta-binomial Wald tests.

Monopoly, DESeq2, edgeR, and `ratioWelch` target the condition difference in the
log2 polysome-to-monosome ratio. FracFixR's Logit and Wald tests target the odds of
one reconstructed fraction against the remainder. Its GLM returns a log2 ratio of
mean reconstructed proportions while testing a binomial condition coefficient.
All are reported under separate estimands. FracFixR fraction-weight recovery is
evaluated in a separate table.

## Run

Install Monopoly with its development dependencies, then install the frozen R
comparators:

```console
python -m pip install -e ".[dev]"
Rscript benchmarks/r/install.R
```

Run the three-replicate software smoke test:

```console
python -m benchmarks.run benchmarks/configs/smoke.json benchmark-results/smoke \
    --rscript Rscript --workers 1
```

`preflight.json` is a small scientific diagnostic. The two `publication_*.json`
files contain the larger count-model and FracFixR grids. They are separate because
FracFixR fits gene-wise models for a related but non-identical estimand and is much
more computationally expensive. Run the smoke test and preflight locally before
submitting either publication grid. See the
[HPC walkthrough](../docs/HPC_BENCHMARKS.md) for a reproducible Slurm workflow.

Runs are resumable. A result directory cannot be resumed after its configuration,
method set, or benchmark source changes. Each run records seeds, configurations,
software versions, simulation hashes, result hashes, failures, and runtime.
The frozen R versions in `r/requirements.tsv` are enforced before any replicate is
run. Completed replicate parts are written atomically, and progress is printed at
approximately one-percent intervals. A missing or empty R result is retried once.
If a replicate still fails, the runner reports its scenario and replicate, retains
all other completed parts, and exits after the remaining submitted work finishes.
Resume an interrupted run with the identical command and output directory. Never
run multiple benchmark processes against the same output directory.

## Outputs

- `manifest.json`: run identity, environment, hashes, and completion state.
- `seedManifest.tsv`: one immutable configuration and seed per replicate.
- `replicateResults.tsv`: operating characteristics for each method and estimand.
- `metricSummary.tsv`: means and Monte Carlo standard errors across replicates.
- `benchmarkAssessment.tsv`: prespecified pass/fail calibration checks.
- `fractionRecovery.tsv`: FracFixR fraction-weight recovery metrics.
- `parts/`: atomic replicate outputs used for resumption and auditing.

The smoke and preflight configurations are engineering checks, not validation
evidence. Publication claims must use the frozen publication grids and report every
prespecified scenario, including failures.
