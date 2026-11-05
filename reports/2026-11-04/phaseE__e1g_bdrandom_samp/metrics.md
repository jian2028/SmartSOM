# e1g_bdrandom_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T142205Z-e1g_bdrandom_samp_seed101-896e35c456 | 20 | 19 | 22.579 | 2797 | 132.981 | 34.150 | 0.453 | 0.725 | 0.112 |
| 20261104T142448Z-e1g_bdrandom_samp_seed102-af0f5d764b | 20 | 19 | 24.537 | 2563.895 | 83.175 | 22.700 | 0.487 | 0.714 | 0.115 |
| 20261104T142736Z-e1g_bdrandom_samp_seed103-2cf6a0c80b | 20 | 19 | 23.800 | 2648.684 | 88.870 | 27.750 | 0.485 | 0.714 | 0.113 |
| 20261104T142906Z-e1g_bdrandom_samp_seed104-6f908f3fec | 20 | 18 | 23.461 | 2637.667 | 74.435 | 22.900 | 0.465 | 0.709 | 0.096 |
| 20261104T152154Z-e1g_bdrandom_samp_seed105-3bfa95b219 | 20 | 19 | 23.454 | 2686.947 | 157.974 | 34.600 | 0.488 | 0.731 | 0.115 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T142205Z-e1g_bdrandom_samp_seed101-896e35c456 | 0.139 | 0.115 | 0.057 | 0.138 |
| 20261104T142448Z-e1g_bdrandom_samp_seed102-af0f5d764b | 0.174 | 0.052 | 0.168 | 0.192 |
| 20261104T142736Z-e1g_bdrandom_samp_seed103-2cf6a0c80b | 0.148 | 0.100 | 0.124 | 0.080 |
| 20261104T142906Z-e1g_bdrandom_samp_seed104-6f908f3fec | 0.079 | 0.133 | 0.179 | 0.401 |
| 20261104T152154Z-e1g_bdrandom_samp_seed105-3bfa95b219 | 0.142 | 0.208 | 0.123 | 0.093 |
