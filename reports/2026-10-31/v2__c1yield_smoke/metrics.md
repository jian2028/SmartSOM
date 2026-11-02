# c1yield_smoke (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T025023Z-c1yield_smoke-755cada5c2 | 1 | 1 | 24.971 | 2563 | 81.641 | 28 | 0.411 | 0.699 | 0.147 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T025023Z-c1yield_smoke-755cada5c2 | 0.710 | 0.591 | 0.700 | 0.288 |
