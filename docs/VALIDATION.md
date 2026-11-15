# Validation scope

Version 0.1.0 is tested as a general paired modeling implementation, not as a claim
of universally calibrated small-sample inference.

The release suite verifies:

- strict alignment and immutability of all inputs;
- rejection of malformed counts and metadata;
- full-rank paired condition-by-fraction designs;
- independent within-fraction normalization and explicit joint normalization;
- participant and gene ordering invariance;
- expected injected-effect direction;
- sign reversal under reversed contrasts and reference conditions;
- stable result and diagnostic schemas;
- complete provenance and input fingerprints;
- deterministic file naming and overwrite behavior; and
- wheel and source-distribution construction.

The numerical backend is PyDESeq2. Monopoly tests its own design construction,
normalization orchestration, contrast vectors, validation, diagnostics, and output
contracts rather than duplicating PyDESeq2's internal test suite.

The release uses a shared gene-wise dispersion across input, monosome, and polysome
fractions. A successful fit does not establish that adjusted P values are
calibrated for every protocol, dispersion pattern, dependence structure, or sample
size. Study authors remain responsible for design provenance, fraction-level
variability checks, sensitivity analyses, and appropriately limited claims.

The clean public benchmark under `benchmarks/` is separate from the unit suite. Its
smoke and preflight configurations validate execution only. Statistical claims must
come from the frozen publication configurations, which retain null failures and
compare only matching estimands. See [BENCHMARKS.md](BENCHMARKS.md).
