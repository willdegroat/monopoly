# Benchmark design

## Purpose

The public benchmark tests implementation agreement, statistical operating
characteristics, and compositional reconstruction. Real-data discovery counts are
not validation because the true effects are unknown.

## Simulation

Each participant contributes input, monosome, and polysome libraries. A latent
participant-specific transcript abundance is modified by transcriptional effects.
An allocation probability divides an explicitly defined associated fraction between
monosome and polysome pools. Redistribution changes the log2 polysome-to-monosome
allocation odds without changing input abundance. Counts are independently sampled
from negative-binomial distributions conditional on their latent means.

This construction satisfies mass balance before sampling: input abundance is
partitioned into monosome, polysome, and unobserved components. It is consequently
compatible with FracFixR's compositional objective while retaining known
condition-by-fraction effects for count-model evaluation.

The frozen grids vary participant count, near-Poisson and overdispersed counts,
equal and unequal fraction dispersions, gene-level dispersion variation,
transcription-only changes, redistribution effect size, and directional asymmetry.
Effects are assigned independently of baseline abundance. Balanced positive and
negative effects expose directional bias; a positive-only scenario is retained as
an asymmetric stress test.

## Estimands

Monopoly, R DESeq2, edgeR, and the ratio test estimate

```text
condition difference in log2(polysome / monosome).
```

FracFixR's native Logit and Wald single-type tests report a different quantity:

```text
condition difference in log2 odds(fraction / remainder).
```

The FracFixR GLM, Logit, and beta-binomial Wald tests are all retained. FracFixR
documents GLM as most powerful, Logit as a faster alternative, and Wald for
overdispersed counts. In version 1.1.0, the GLM p-value tests a binomial condition
coefficient but its returned `log2FC` is the log2 ratio of group-mean proportions;
the benchmark therefore evaluates that reported effect against proportion-ratio
truth. No FracFixR output is relabeled as a polysome-to-monosome contrast. Effect
bias and interval coverage are compared only with matching simulation truth.
The adapter does not repair or replace native FracFixR calculations. An estimand
with no finite p-values is retained and marked as a method failure rather than
silently omitted.

## Metrics

Each replicate reports type-I error, Benjamini-Hochberg FDR, power, effect bias,
RMSE, interval coverage when a standard error is available, false-sign rate, finite
result rates, failures, and runtime. Power and effect bias are also separated by
positive and negative truth so directional failures cannot cancel. Missing genes
are nonfinite and count as missed alternatives. Type-I error uses finite null
p-values and must be interpreted with the null finite rate.

Replicates, not genes, are the Monte Carlo units. Summary tables report replicate
means and Monte Carlo standard errors. Pooled FDR and power are descriptive
supplements rather than replacements for replicate-level summaries. Pooled FDR is
not assessed under a global null because it equals one whenever any rejection is
made; replicate-mean FDR is the relevant global-null quantity.

FracFixR's estimated monosome, polysome, and lost weights are aligned to known
participant-level fractions. Overall reconstruction is assessed by RMSE, maximum
absolute error, correlation, finite rate, range violations, and simplex error.
Bias, RMSE, and correlation are also reported separately for each fraction;
aggregate bias is omitted because compensating simplex errors force it toward zero.

## Adversarial criteria

The benchmark must retain every frozen scenario. A method is not described as
calibrated in a core scenario if any of the following persist:

- mean type-I error exceeds 0.06 at nominal 0.05;
- mean FDR exceeds 0.06, or pooled FDR exceeds 0.06 in a scenario containing
  alternatives;
- 95% interval coverage falls outside 0.925 to 0.975;
- finite results occur for fewer than 99.5% of genes; or
- absolute effect bias exceeds 0.1 log2 units above the mean-count threshold.

The exported assessment uses the larger absolute bias across positive and negative
alternatives. Metrics that a method does not provide, such as interval coverage
without standard errors, are marked `notEvaluated`; they are never counted as
passes. Power is reported but has no universal pass threshold because it depends on
the prespecified effect size and sample count.

Monopoly and R DESeq2 should agree numerically because they implement the same
working model. Their comparison establishes implementation correctness, not
superiority. edgeR and FracFixR use different estimation machinery and are judged on
their declared targets. No method is removed because it performs poorly.

## Boundaries

The simulator cannot prove performance under every protocol, dependence structure,
or global shift. Fraction-wise normalization cannot identify a protocol-wide change
affecting nearly every gene without spike-ins, measured recovery, or invariant
controls. FracFixR attempts to recover global fraction weights from input and
fractionated libraries; this is evaluated explicitly rather than assumed. Biological
datasets are application tests, not ground truth for calibration.

## References

- Love MI, Huber W, Anders S. [DESeq2](https://doi.org/10.1186/s13059-014-0550-8).
- McCarthy DJ, Chen Y, Smyth GK.
  [edgeR quasi-likelihood](https://doi.org/10.1093/nar/gks042).
- Cleynen A, Ravindran A, Shirokikh NE.
  [FracFixR](https://doi.org/10.1093/bioinformatics/btaf615).
