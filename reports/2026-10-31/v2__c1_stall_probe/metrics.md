# c1_stall_probe (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T000341Z-c1_stall_probe_seed104-0a4998f4cd | 2 | 1 | 20.318 | 2486 | 6.601 | 3.500 | 0.350 | 0.357 | 0.015 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T000341Z-c1_stall_probe_seed104-0a4998f4cd | 0.651 | 0.218 | 0.210 | 0.605 |
