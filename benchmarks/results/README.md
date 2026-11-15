# Publication benchmark results

Complete frozen outputs are archived with the study repository:

<https://github.com/willdegroat/asd-polysome-profiling/tree/main/published/manuscript/benchmarks>

Run `python scripts/restore_results.py` in that repository to unpack and verify
the original count-model grid, FracFixR native-estimand comparisons, and the
additional equal-dispersion scenario. Tables retain failed scientific calibration
criteria and Monte Carlo standard errors. No failed or unfavorable scenario was
removed from the archive.

The original count-model grid contains 6,200 datasets. The added equal-dispersion
configuration contains 200 datasets at 20 participants per condition, matching
the unequal-dispersion scenario's sample size, genes, effects, and random seeds.
Together these comprise 6,400 count-model datasets. FracFixR's 900 datasets use
its corresponding estimands and should be interpreted separately.

The source files are byte-preserved because manifests fingerprint the benchmark
code. The recorded hash is
`sha256:7fb95b9ca4c79a0218b0d5b900db1360741ba2ce077cada9853c8e97e6321c99`.
Reruns use different system loads and should not be expected to reproduce exact
runtimes. All displayed runtime comparisons used the same Amarel node and
52-worker allocation in separate jobs.

```console
python -m benchmarks.run benchmarks/configs/publication_equal_n20.json \
    benchmark-results/equal-n20 --rscript Rscript --workers 52
```

Install the pinned R dependencies described in `docs/HPC_BENCHMARKS.md` first.
