# HPC benchmark walkthrough

The publication configurations are frozen. Do not edit them after a run begins.
Use a distinct output directory for each configuration, and never launch concurrent
processes against the same output directory.

## Environment

The benchmark requires Python 3.11 or later, R 4.6.1, a C/C++ toolchain, zlib
development headers, and enough storage for replicate parts. On a module-based
cluster, load the site-specific Python and R modules before creating the
environments. In a Conda environment, include the `compilers`, `make`,
`pkg-config`, and `zlib` packages.

```console
git clone https://github.com/WillDeGroat/monopoly.git
cd monopoly
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export R_LIBS_USER="$PWD/.r-library"
mkdir -p "$R_LIBS_USER"
Rscript benchmarks/r/install.R
Rscript benchmarks/r/run_comparators.R --versions
```

The final command must report R 4.6.1, FracFixR 1.1.0, DESeq2 1.52.0, and edgeR
4.10.5. The installer stops if these frozen versions are unavailable.

## Local verification

Run the Python tests, the smoke benchmark, and the preflight before requesting a
large allocation:

```console
export MONOPOLY_RSCRIPT="$(command -v Rscript)"
python -m ruff check .
python -m pytest --cov=monopoly
python -m benchmarks.run benchmarks/configs/smoke.json \
    benchmark-results/smoke --rscript "$(command -v Rscript)" --workers 1
python -m benchmarks.run benchmarks/configs/preflight.json \
    benchmark-results/preflight --rscript "$(command -v Rscript)" --workers 1
```

Both manifests must end with `"status": "complete"`. Method failures and failed
calibration criteria in preflight are scientific outputs, not reasons to delete a
scenario or modify the method.

## Slurm execution

The following is a conservative single-node template. Adjust the account, partition,
memory, and wall time to the cluster. Save it as `benchmark.slurm` outside the
repository or in a local ignored directory.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=monopoly-benchmark
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=monopoly-%j.log

set -euo pipefail

cd /path/to/monopoly
source .venv/bin/activate

export R_LIBS_USER="$PWD/.r-library"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${CONFIG:?Set CONFIG to a frozen benchmark JSON file}"
: "${OUTPUT:?Set OUTPUT to a unique result directory}"

python -m benchmarks.run "$CONFIG" "$OUTPUT" \
    --rscript "$(command -v Rscript)" \
    --workers "$SLURM_CPUS_PER_TASK"
```

Start with fewer workers if the preflight shows high peak memory. Each worker runs
one replicate and invokes R comparators serially; limiting numerical-library threads
prevents hidden oversubscription.

Submit the two publication grids to different output directories:

```console
sbatch --export=ALL,CONFIG=benchmarks/configs/publication_count_models.json,OUTPUT=benchmark-results/publication-count-models benchmark.slurm
sbatch --export=ALL,CONFIG=benchmarks/configs/publication_fracfixr.json,OUTPUT=benchmark-results/publication-fracfixr benchmark.slurm
```

Completed replicate parts are atomic and remain usable after a timeout or worker
failure. Missing or empty R output is retried once. If a replicate still fails, its
scenario and replicate are printed while other submitted results continue to be
saved. Resubmit the identical command to the identical output directory to continue.
If the code, configuration, or method set changes, the runner rejects the resume;
start a new output directory rather than mixing builds.

## Completion checks

A complete count-model run contains 6,200 rows in `seedManifest.tsv`, 12,400
part files, and 24,800 rows in `replicateResults.tsv`. A complete FracFixR run
contains 900 seed rows, 1,800 part files, 6,300 rows in
`replicateResults.tsv`, and 900 rows in `fractionRecovery.tsv`. Confirm these
counts against `manifest.json`; each replicate has one gene-metric part and one
fraction-recovery part, including empty fraction parts when FracFixR is not run.

Before interpretation:

1. Confirm that each manifest reports `"status": "complete"`.
2. Preserve every scenario and every `methodError`.
3. Review `benchmarkAssessment.tsv` without treating unevaluated metrics as passes.
4. Archive the complete result directories, including `parts/` and the job logs.
5. Record the Git commit and retain the manifest hashes with the manuscript outputs.

Do not compare FracFixR's native single-fraction tests as if they estimated the
polysome-to-monosome interaction. The matching estimands and prespecified criteria
are defined in [BENCHMARKS.md](BENCHMARKS.md).
