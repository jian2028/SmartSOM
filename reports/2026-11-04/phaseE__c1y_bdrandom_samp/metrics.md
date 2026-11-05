# c1y_bdrandom_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T210159Z-c1y_bdrandom_samp_seed101-ac5f2b63b6 | 20 | 17 | 22.271 | 2734.118 | 109.683 | 29.550 | 0.449 | 0.697 | 0.115 |
| 20261104T210159Z-c1y_bdrandom_samp_seed102-7b940db6d0 | 20 | 17 | 21.888 | 2792.353 | 179.357 | 35.600 | 0.444 | 0.704 | 0.123 |
| 20261104T210159Z-c1y_bdrandom_samp_seed103-768c46b01d | 20 | 18 | 21.631 | 2873.056 | 146.009 | 35.100 | 0.439 | 0.701 | 0.109 |
| 20261104T210159Z-c1y_bdrandom_samp_seed104-3cccc67e12 | 20 | 18 | 22.122 | 2804.889 | 149.563 | 34.600 | 0.448 | 0.714 | 0.115 |
| 20261104T210159Z-c1y_bdrandom_samp_seed105-0b1c0c3c49 | 20 | 20 | 22.043 | 2920.800 | 156.593 | 34.700 | 0.448 | 0.725 | 0.112 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T210159Z-c1y_bdrandom_samp_seed101-ac5f2b63b6 | 0.238 | 0.066 | 0.240 | 0.098 |
| 20261104T210159Z-c1y_bdrandom_samp_seed102-7b940db6d0 | 0.052 | 0.062 | 0.165 | 0.117 |
| 20261104T210159Z-c1y_bdrandom_samp_seed103-768c46b01d | 0.286 | 0.157 | 0.441 | 0.628 |
| 20261104T210159Z-c1y_bdrandom_samp_seed104-3cccc67e12 | 0.221 | 0.252 | 0.241 | 0.140 |
| 20261104T210159Z-c1y_bdrandom_samp_seed105-0b1c0c3c49 | 0.365 | 0.062 | 0.335 | 0.136 |
