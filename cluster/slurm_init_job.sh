#!/usr/bin/env bash
#SBATCH --job-name=reg-bench-init
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-init-%j.out
#SBATCH --error=slurm-init-%j.err

set -euo pipefail

: "${BENCHMARK_PAIRS:?Set BENCHMARK_PAIRS to the pair CSV path.}"
: "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG to the YAML config path.}"
: "${BENCHMARK_OUT:?Set BENCHMARK_OUT to the output directory.}"

DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-1}"
ALREADY_SKULL_STRIPPED="${ALREADY_SKULL_STRIPPED:-0}"

args=(
  --pairs "$BENCHMARK_PAIRS"
  --config "$BENCHMARK_CONFIG"
  --out-dir "$BENCHMARK_OUT"
  --downsample-factor "$DOWNSAMPLE_FACTOR"
  --initialize-only
)

if [ "$ALREADY_SKULL_STRIPPED" = "1" ]; then
  args+=(--already-skull-stripped)
fi

registration-benchmark "${args[@]}"
