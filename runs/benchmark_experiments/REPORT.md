# Benchmark Experiment Report

- Benchmarks directory: `configs/benchmarks`
- Runs directory: `runs/benchmark_experiments`
- Max iterations per run: `3`
- Passes: `2`

## Result files

- CSV: `benchmark_results.csv`
- Summary JSON: `benchmark_summary.json`
- Plot: `iterations_by_run.png`
- Plot: `solved_status_by_run.png`
- Plot: `adaptive_curation_outcomes.png`

## Runs

| Problem | Pass | Solved | Iterations | Objective | Added | Merged | Rejected | Return code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| assignment_small | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| changeover_sequence | 1 | ✅ | 1 | - | 1 | 1 | 0 | 0 |
| changeover_small | 1 | ❌ | 3 | - | 1 | 0 | 0 | 0 |
| eligibility_sparse | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| flow_shop_small | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| job_shop_tiny | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| parallel_machine_assignment | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| single_machine_tardiness | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| tardiness_small | 1 | ❌ | 3 | - | 2 | 0 | 2 | 0 |
| worker_capacity_multi_skill | 1 | ❌ | 3 | - | 0 | 0 | 0 | 0 |
| worker_capacity_small | 1 | ✅ | 0 | - | 0 | 0 | 0 | 0 |
| changeover_small | 2 | ✅ | 1 | - | 2 | 0 | 1 | 0 |
| tardiness_small | 2 | ✅ | 1 | - | 0 | 0 | 2 | 0 |
| worker_capacity_multi_skill | 2 | ❌ | 3 | - | 0 | 0 | 0 | 0 |