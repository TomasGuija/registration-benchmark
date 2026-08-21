# Cluster benchmarking

The benchmark's cluster layer assigns one independent image pair to each SLURM
array task. The core pipeline remains scheduler-independent.

## SLURM array

Install the repository in the environment used by compute nodes, then export
absolute paths visible from every node:

```bash
conda activate dipy-ants-benchmarks

export BENCHMARK_PAIRS=/shared/data/oasis2_pairs.csv
export BENCHMARK_CONFIG="$PWD/configs/syn_cc_default.yaml"
export BENCHMARK_OUT=/shared/results/run_001
export DOWNSAMPLE_FACTOR=2
export USE_CUDA=0
export BENCHMARK_ARRAY=0-99

bash cluster/submit_slurm.sh
```

The array range is zero-based: `0-99` runs the first 100 CSV rows. Each task
writes:

```text
BENCHMARK_OUT/<pair_id>/sample_result.json
```

The submission script queues a small initialization job and submits the array
with an `afterok` dependency. The array therefore starts only if
`BENCHMARK_OUT/run_metadata.json` was created or validated successfully. Pair
jobs also check that the file exists, avoiding concurrent metadata creation.
Local runs do not need this extra job because one process creates or validates
the metadata before registration begins.

The template requests the same number of scheduler CPUs as
`registration.num_threads` in the default configuration. Adjust both together
when benchmarking another thread count. Also adapt partition, memory, time,
module loading, and environment activation for the target cluster.

SynthSeg uses the CPU by default. To use CUDA, set `USE_CUDA=1` and add the GPU
resource request required by the target cluster, such as `#SBATCH --gres=gpu:1`,
to `slurm_pair_job.sh`.

## Collect results

After all jobs finish, collect exactly the expected first `N` rows:

```bash
registration-collect \
  --out-dir /shared/results/run_001 \
  --out-json /shared/results/run_001/benchmark_results.json \
  --pairs /shared/data/oasis2_pairs.csv \
  --n 100
```

This mode fails if any expected result is missing. To collect every available
pair result instead, omit `--pairs` and `--n`:

```bash
registration-collect \
  --out-dir /shared/results/run_001 \
  --out-json /shared/results/run_001/benchmark_results.json
```

The collector also rejects mixed run metadata, preventing results produced with
different configurations or software environments from being summarized as one
benchmark.

For PBS/Torque or another scheduler, retain the `registration-benchmark
--pair-index` command and replace only the scheduler wrapper.
