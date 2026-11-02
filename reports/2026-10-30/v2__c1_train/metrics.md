# c1_train (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T005724Z-c1_train-a0bdce4b1c | 100 | 53 | 17.728 | 2559.717 | 32.163 | 11.930 | 0.371 | 0.469 | 0.101 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T005724Z-c1_train-a0bdce4b1c | 0.381 | 0.181 | 0.291 | 0.309 |
