# DIPY–ANTs registration benchmarks

<p align="center">
  <a href="https://dipy.org/">
    <img src="https://raw.githubusercontent.com/dipy/dipy/master/doc/_static/images/logos/dipy-logo.png" alt="DIPY logo" width="420">
  </a>
</p>

<p align="center">
  <a href="https://github.com/dipy/dipy">DIPY on GitHub</a> ·
  <a href="https://dipy.org/">DIPY website</a>
</p>

*Documentation and API last reviewed: 2026-08-26*

Reproducible, dataset-agnostic benchmarks comparing DIPY and ANTsPy nonlinear
image registration. The initial benchmark evaluates SyN-only registration with
cross-correlation (CC) and mutual information (MI).

This repository contains benchmark code only. DIPY and ANTsPy are installed as
dependencies; no implementation from either project is vendored here.

## What the benchmark does

For every fixed/moving image pair, the pipeline:

1. reslices both images by the same optional downsampling factor;
2. skull-strips them with DIPY SynthSeg, unless they are already stripped;
3. fills mask holes and retains the largest connected component;
4. rigidly prealigns the moving image with DIPY;
5. runs DIPY SyN and ANTsPy `SyNOnly` from the same prealigned inputs;
6. warps label maps with nearest-neighbor interpolation; and
7. evaluates NMI, NCC, Dice, Jaccard, and SyN runtime.

The framework consumes a dataset-independent CSV:

```text
pair_id,fixed_path,moving_path,fixed_label_path,moving_label_path
```

Only `fixed_path` and `moving_path` are required. Dataset adapters create this
CSV; raw datasets and generated results are deliberately excluded from Git.

## Latest benchmark results


### Monomodal registration

The monomodal CC benchmark uses **— samples** from **OASIS-2**. Dataset
preparation code:
[`datasets/oasis2/make_pairs.py`](datasets/oasis2/make_pairs.py).

| Registration stage | Mean NCC | Average SyN time (s) |
| ------------------ | ------: | ---------------------: |
| Rigidly prealigned | — | Not applicable |
| DIPY SyN | — | — |
| ANTs SyN | — | — |

### Multimodal registration

The multimodal MI benchmark uses **100 inter-patient pairs** from the
annotated **MRBrainS** training subjects. The inputs combine FLAIR, IR, and T1 images.

| Registration stage | Mean NMI | Average SyN time (s) |
| ------------------ | -------: | ---------------------: |
| Rigidly prealigned | 1.0173 | Not applicable |
| DIPY SyN | 1.0382 | 69.22 |
| ANTs SyN | 1.0348 | 113.64 |

## Reproducible installation

The reference environment uses Python 3.13 and pins all numerical dependencies,
including the exact DIPY development revision on which the benchmark was
validated:

```bash
conda env create -f environment.yml
conda activate dipy-ants-benchmarks
```

For inputs that already include label maps and skull-stripped images, the base
environment is sufficient. Install the optional SynthSeg dependency when the
pipeline must generate masks or labels:

```bash
python -m pip install -e ".[synthseg]"
```

The benchmark records software versions, platform details, configuration, and
preprocessing settings in its result JSON.

## Prepare a dataset

OASIS-2 is included as an example adapter (code and documentation only):

```bash
python datasets/oasis2/make_pairs.py \
  --root /path/to/OAS2_RAW_PART1/OAS2_RAW_PART1 \
  --out data/oasis2_pairs.csv
```

See [`datasets/oasis2/README.md`](datasets/oasis2/README.md). Contributors can
add other adapters without changing the benchmark pipeline.

## Run locally

```bash
registration-benchmark \
  --pairs data/oasis2_pairs.csv \
  --config configs/syn_cc_default.yaml \
  --out-dir outputs/oasis2_syn_cc_ds2 \
  --downsample-factor 2 \
  --n 5
```

Use `--already-skull-stripped` when appropriate and `--use-cuda` to run
SynthSeg on CUDA. `--n N` deterministically selects the first `N` CSV rows.

Each pair writes `sample_result.json`. A local multi-pair run additionally
writes `benchmark_results.json` with progressive aggregate statistics.

SyN timing measures registration through creation of the warped intensity
image. It excludes input loading, label warping, and output writing. See
[`Timing boundary`](docs/fair_comparison.md#timing-boundary) for details.

The output root also contains `run_metadata.json`, which records the pair CSV,
pair limit, and preprocessing settings. Reusing an output directory with the
same contract is allowed: cached prepared images are reused and existing
registration results are overwritten, with explicit warnings. A changed pair
CSV path, pair limit, downsampling factor, or skull-stripping mode is rejected;
use a new output directory for that run. Local runs create or validate this file
automatically.

## Run on a cluster

The SLURM array template assigns one zero-based CSV pair index to each task:

```bash
export BENCHMARK_PAIRS=/absolute/path/to/pairs.csv
export BENCHMARK_CONFIG="$PWD/configs/syn_cc_default.yaml"
export BENCHMARK_OUT=/absolute/path/to/results/run_001
export DOWNSAMPLE_FACTOR=2
export BENCHMARK_ARRAY=0-99

bash cluster/submit_slurm.sh
```

The submission script queues a small metadata job followed by the pair array,
which runs only if metadata creation or validation succeeds.

See [`cluster/README.md`](cluster/README.md) for collection and scheduler
configuration.

## Fairness and interpretation

DIPY and ANTs do not expose identical SyN algorithms or parameterizations.
[`docs/fair_comparison.md`](docs/fair_comparison.md) documents the mappings,
known mismatches, and assumptions. Results should always be reported with the
configuration, dependency versions, hardware, thread count, preprocessing, and
dataset-pair selection.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Project context

This repository was developed by
[Tomás Guija Valiente](https://github.com/TomasGuija) as part of
[Google Summer of Code 2026 with DIPY](https://summerofcode.withgoogle.com/programs/2026/projects/wtuoFjpU).
It contains the registration benchmarking work developed alongside the project.

Questions, suggestions, and bug reports are welcome through this repository's
GitHub issues. You can also contact me through my GitHub profile.
