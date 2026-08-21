#!/usr/bin/env bash
#SBATCH --job-name=reg-bench
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

: "${BENCHMARK_PAIRS:?Set BENCHMARK_PAIRS to the pair CSV path.}"
: "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG to the YAML config path.}"
: "${BENCHMARK_OUT:?Set BENCHMARK_OUT to the output directory.}"

DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-1}"
USE_CUDA="${USE_CUDA:-0}"
ALREADY_SKULL_STRIPPED="${ALREADY_SKULL_STRIPPED:-0}"

if [ ! -f "$BENCHMARK_OUT/run_metadata.json" ]; then
  echo "Missing $BENCHMARK_OUT/run_metadata.json. Submit with cluster/submit_slurm.sh." >&2
  exit 1
fi

args=(
  --pairs "$BENCHMARK_PAIRS"
  --config "$BENCHMARK_CONFIG"
  --out-dir "$BENCHMARK_OUT"
  --downsample-factor "$DOWNSAMPLE_FACTOR"
  --pair-index "$SLURM_ARRAY_TASK_ID"
)

if [ "$USE_CUDA" = "1" ]; then
  args+=(--use-cuda)
fi
if [ "$ALREADY_SKULL_STRIPPED" = "1" ]; then
  args+=(--already-skull-stripped)
fi

registration-benchmark "${args[@]}"
