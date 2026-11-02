# c1y_train (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T092118Z-c1y_train-d4c586a752 | 100 | 92 | 23.467 | 2657.750 | 94.712 | 27.760 | 0.475 | 0.728 | 0.116 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T092118Z-c1y_train-d4c586a752 | 0.240 | 0.136 | 0.330 | 0.243 |
