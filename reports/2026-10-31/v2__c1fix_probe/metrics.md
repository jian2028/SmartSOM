# c1fix_probe (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T001333Z-c1fix_probe_seed104-74d5d6ff04 | 2 | 1 | 20.059 | 2489 | 6.925 | 6.500 | 0.353 | 0.463 | 0.015 |
| 20261031T002046Z-c1fix_probe_seed104-373c914903 | 2 | 1 | 20.059 | 2489 | 6.925 | 6.500 | 0.353 | 0.463 | 0.015 |
| 20261031T002924Z-c1fix_probe_seed104-525f098514 | 2 | 0 | 14.648 | – | 5.572 | 3 | 0.263 | 0.284 | 0.044 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T001333Z-c1fix_probe_seed104-74d5d6ff04 | 0.627 | 0.211 | 0.183 | 0.590 |
| 20261031T002046Z-c1fix_probe_seed104-373c914903 | 0.627 | 0.211 | 0.183 | 0.590 |
| 20261031T002924Z-c1fix_probe_seed104-525f098514 | 0.485 | 0.178 | 0.130 | 0.471 |
