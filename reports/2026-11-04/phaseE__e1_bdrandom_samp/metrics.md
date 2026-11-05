# e1_bdrandom_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T154026Z-e1_bdrandom_samp_seed101-c0846593a6 | 20 | 18 | 22.328 | 2783.944 | 124.058 | 28.800 | 0.458 | 0.711 | 0.109 |
| 20261104T154509Z-e1_bdrandom_samp_seed102-06a3abc555 | 20 | 19 | 24.095 | 2616.158 | 83.911 | 26.050 | 0.486 | 0.699 | 0.115 |
| 20261104T155132Z-e1_bdrandom_samp_seed103-d638390660 | 20 | 20 | 23.798 | 2696.700 | 118.983 | 34.350 | 0.483 | 0.715 | 0.124 |
| 20261104T155230Z-e1_bdrandom_samp_seed104-daeced4cfa | 20 | 20 | 24.778 | 2595.200 | 62.959 | 22.700 | 0.503 | 0.722 | 0.116 |
| 20261104T155340Z-e1_bdrandom_samp_seed105-96f4ab3e17 | 20 | 18 | 24.182 | 2546.111 | 73.759 | 23.400 | 0.492 | 0.711 | 0.115 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T154026Z-e1_bdrandom_samp_seed101-c0846593a6 | 0.157 | 0.106 | 0.067 | 0.124 |
| 20261104T154509Z-e1_bdrandom_samp_seed102-06a3abc555 | 0.364 | 0.131 | 0.052 | 0.113 |
| 20261104T155132Z-e1_bdrandom_samp_seed103-d638390660 | 0.095 | 0.146 | 0.078 | 0.111 |
| 20261104T155230Z-e1_bdrandom_samp_seed104-daeced4cfa | 0.097 | 0.088 | 0.080 | 0.174 |
| 20261104T155340Z-e1_bdrandom_samp_seed105-96f4ab3e17 | 0.091 | 0.066 | 0.093 | 0.051 |
