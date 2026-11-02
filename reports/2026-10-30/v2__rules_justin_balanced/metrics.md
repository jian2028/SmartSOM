# rules_justin_balanced (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T000624Z-rules_justin_balanced-e85c5fd5b0 | 20 | 19 | 25.693 | 2446 | 29.975 | 8.950 | 0.548 | 0.470 | – |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T000624Z-rules_justin_balanced-e85c5fd5b0 | 0.100 | 0.323 | 0.324 | 0.056 |
