# rules_justin_low (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T000621Z-rules_justin_low-c7ce13c743 | 20 | 19 | 24.846 | 2541.895 | 60.594 | 12.500 | 0.532 | 0.447 | – |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T000621Z-rules_justin_low-c7ce13c743 | 0.074 | 0.318 | 0.266 | 0.069 |
