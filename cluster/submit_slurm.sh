#!/usr/bin/env bash

set -euo pipefail

: "${BENCHMARK_PAIRS:?Set BENCHMARK_PAIRS to the pair CSV path.}"
: "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG to the YAML config path.}"
: "${BENCHMARK_OUT:?Set BENCHMARK_OUT to the output directory.}"
: "${BENCHMARK_ARRAY:?Set BENCHMARK_ARRAY to a zero-based range, for example 0-99.}"

init_submission=$(sbatch --parsable cluster/slurm_init_job.sh)
init_job_id="${init_submission%%;*}"

array_submission=$(
  sbatch \
    --parsable \
    --dependency="afterok:$init_job_id" \
    --kill-on-invalid-dep=yes \
    --array="$BENCHMARK_ARRAY" \
    cluster/slurm_pair_job.sh
)
array_job_id="${array_submission%%;*}"

echo "Submitted metadata job $init_job_id and dependent array $array_job_id."
