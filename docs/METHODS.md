# Statistical method

## Estimand

For gene *g*, participant *i*, and fraction *f* in input (*I*), monosome (*M*), or
polysome (*P*), let `Y[g, i, f]` denote the observed count and `mu[g, i, f]` its
mean. The primary estimand is the condition difference in the log2
polysome-to-monosome ratio:

```text
theta[g] = E[log2(mu[g, i, P]) - log2(mu[g, i, M]) | test]
         - E[log2(mu[g, i, P]) - log2(mu[g, i, M]) | reference].
```

Polysome-to-input and monosome-to-input contrasts are defined analogously. These are
relative fraction-association estimands, not absolute protein-synthesis rates.

## Mean model

Monopoly uses the working model

```text
Y[g, i, f] ~ NB(mu[g, i, f], alpha[g])

log(mu[g, i, f]) = log(s[i, f])
                  + participant[g, i]
                  + fraction[g, f]
                  + condition[i] * fraction[g, f]
                  + covariates[g, i, f].
```

`s[i, f]` is a size factor and `alpha[g]` is one moderated dispersion per gene
shared across fractions. Condition is constant within participant and is absorbed by
the participant fixed effect; the condition-by-fraction terms remain estimable.
Optional covariates must vary within participants and preserve a full-rank design
with positive residual degrees of freedom.

The primary contrast subtracts the monosome interaction coefficient from the
polysome interaction coefficient. Positive values indicate a larger normalized
polysome-to-monosome ratio in the test condition.

## Normalization

The default estimates DESeq2 median-of-ratios size factors independently within
input, monosome, and polysome libraries. Separate fraction constants cancel from the
difference-in-differences estimand. `poscounts`, `iterative`, and validated control
genes are supported through PyDESeq2.

Joint normalization is retained as a sensitivity option. It assumes that all three
fractions can share one reference composition and should not be selected merely
because it produces more discoveries.

No internal normalization can recover absolute fraction yields or a global shift
affecting most genes in one direction. Such targets require appropriately introduced
spike-ins, fraction-recovery measurements, or defensible invariant controls.

## Inference and diagnostics

PyDESeq2 supplies dispersion moderation, coefficient estimation, Wald statistics,
and Benjamini-Hochberg adjustment. The released model records
`dispersionModel = 'shared'`, `inferenceModel = 'asymptoticWald'`, and
`inferenceScope = 'workingModel'` in provenance.

The shared-dispersion assumption is most plausible when fraction-specific
mean-variance relationships are similar. It is not guaranteed by paired sampling.
Users should inspect fraction-level variability and interpret small-study adjusted
P values cautiously. The shared-dispersion model is the only inference model in
version 0.1.

Cook's distances are retained, but automatic Cook's filtering and replacement are
disabled. With participant fixed effects, every participant-fraction library is a
unique modeled cell, so automatic replacement lacks exchangeable replicates at that
cell.

## Scope

The model assumes:

- identical participants and genes across all three count matrices;
- exactly two conditions with at least two participants in each;
- matched fractions from each participant;
- conditionally independent libraries after modeled effects;
- one adequate moderated dispersion per gene across fractions; and
- a normalization reference set that is not globally asymmetric within fractions.

Family structure, repeated cell lines from one donor, unmeasured batch effects, and
absolute fraction recovery require study-specific design information that count
matrices alone cannot supply.

## References

- Love MI, Huber W, Anders S. [Moderated estimation of fold change and dispersion
  for RNA-seq data with DESeq2](https://doi.org/10.1186/s13059-014-0550-8).
- Muzellec B, Teleńczuk M, Cabeli V, Andreux M. [PyDESeq2: a Python package for bulk
  RNA-seq differential expression
  analysis](https://doi.org/10.1093/bioinformatics/btad547).
- Love MI et al. [tximport](https://doi.org/10.18129/B9.bioc.tximport).
- Heyer EE, Moore MJ. [Redefining the translational status of 80S
  monosomes](https://doi.org/10.1016/j.cell.2016.01.003).
