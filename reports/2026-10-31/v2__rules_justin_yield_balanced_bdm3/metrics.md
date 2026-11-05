# rules_justin_yield_balanced_bdm3 (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T085733Z-rules_justin_yield_balanced_bdm3-3b32aa454e | 20 | 19 | 23.762 | 2664.737 | 66.821 | 18.100 | 0.511 | 0.602 | 0.093 |
| 20261105T005711Z-rules_justin_yield_balanced_bdm3_seed102-75882c498f | 20 | 19 | 23.762 | 2664.737 | 66.821 | 18.100 | 0.511 | 0.602 | 0.093 |
| 20261105T005712Z-rules_justin_yield_balanced_bdm3_seed103-f84eae534b | 20 | 19 | 23.762 | 2664.737 | 66.821 | 18.100 | 0.511 | 0.602 | 0.093 |
| 20261105T021801Z-rules_justin_yield_balanced_bdm3_seed104-a4f22b4a73 | 20 | 19 | 23.762 | 2664.737 | 66.821 | 18.100 | 0.511 | 0.602 | 0.093 |
| 20261105T021813Z-rules_justin_yield_balanced_bdm3_seed105-19bd24bdbe | 20 | 19 | 23.762 | 2664.737 | 66.821 | 18.100 | 0.511 | 0.602 | 0.093 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T085733Z-rules_justin_yield_balanced_bdm3-3b32aa454e | 0.095 | 0.244 | 0.130 | 0.095 |
| 20261105T005711Z-rules_justin_yield_balanced_bdm3_seed102-75882c498f | 0.095 | 0.244 | 0.130 | 0.095 |
| 20261105T005712Z-rules_justin_yield_balanced_bdm3_seed103-f84eae534b | 0.095 | 0.244 | 0.130 | 0.095 |
| 20261105T021801Z-rules_justin_yield_balanced_bdm3_seed104-a4f22b4a73 | 0.095 | 0.244 | 0.130 | 0.095 |
| 20261105T021813Z-rules_justin_yield_balanced_bdm3_seed105-19bd24bdbe | 0.095 | 0.244 | 0.130 | 0.095 |
