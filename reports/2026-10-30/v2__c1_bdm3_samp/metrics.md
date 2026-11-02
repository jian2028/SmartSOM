# c1_bdm3_samp (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T155446Z-c1_bdm3_samp_seed101-fab963dbf6 | 20 | 9 | 15.578 | 2700.778 | 49.895 | 14.250 | 0.344 | 0.386 | 0.051 |
| 20261030T155446Z-c1_bdm3_samp_seed102-352f96324e | 20 | 6 | 13.996 | 2678.333 | 50.613 | 9.350 | 0.297 | 0.509 | 0.165 |
| 20261030T155447Z-c1_bdm3_samp_seed104-706b4c2879 | 20 | 6 | 14.270 | 2531 | 42.177 | 12.650 | 0.311 | 0.425 | 0.121 |
| 20261030T155447Z-c1_bdm3_samp_seed105-e4a46596e9 | 20 | 6 | 15.060 | 2470.667 | 37.564 | 10.050 | 0.325 | 0.461 | 0.141 |
| 20261030T155448Z-c1_bdm3_samp_seed103-ee5c537744 | 20 | 13 | 18.557 | 2707.462 | 69.365 | 21 | 0.398 | 0.458 | 0.066 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T155446Z-c1_bdm3_samp_seed101-fab963dbf6 | 0.371 | 0.379 | 0.106 | 0.143 |
| 20261030T155446Z-c1_bdm3_samp_seed102-352f96324e | 0.109 | 0.077 | 0.205 | 0.183 |
| 20261030T155447Z-c1_bdm3_samp_seed104-706b4c2879 | 0.450 | 0.177 | 0.150 | 0.431 |
| 20261030T155447Z-c1_bdm3_samp_seed105-e4a46596e9 | 0.358 | 0.038 | 0.181 | 0.120 |
| 20261030T155448Z-c1_bdm3_samp_seed103-ee5c537744 | 0.497 | 0.211 | 0.266 | 0.548 |
