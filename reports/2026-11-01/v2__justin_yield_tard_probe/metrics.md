# justin_yield_tard_probe (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261101T052131Z-justin_yield_tard_probe-7b3444a4e0 | 2 | 2 | 23.845 | 2684 | 32.633 | 9 | 0.498 | 0.580 | 0.090 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261101T052131Z-justin_yield_tard_probe-7b3444a4e0 | 0.080 | 0.291 | 0.244 | 0.063 |
