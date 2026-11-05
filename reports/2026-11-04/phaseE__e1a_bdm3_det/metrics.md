# e1a_bdm3_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T021844Z-e1a_bdm3_det_seed101-f9cd3ae515 | 20 | 19 | 24.297 | 2618.842 | 148.510 | 21.100 | 0.384 | 0.673 | 0.135 |
| 20261104T021955Z-e1a_bdm3_det_seed102-714203a25c | 20 | 18 | 19.961 | 3142.111 | 155.281 | 34 | 0.476 | 0.656 | 0.079 |
| 20261104T023716Z-e1a_bdm3_det_seed103-f21e96d205 | 20 | 15 | 21.648 | 2752.333 | 134.580 | 29 | 0.431 | 0.649 | 0.124 |
| 20261104T025951Z-e1a_bdm3_det_seed104-0a1b070df9 | 20 | 19 | 24.136 | 2637.211 | 101.433 | 22.800 | 0.479 | 0.706 | 0.110 |
| 20261104T040839Z-e1a_bdm3_det_seed105-037eeaf732 | 20 | 18 | 23.567 | 2659.444 | 101.958 | 17.750 | 0.370 | 0.674 | 0.122 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T021844Z-e1a_bdm3_det_seed101-f9cd3ae515 | 0.125 | 0.108 | 0.059 | 0.076 |
| 20261104T021955Z-e1a_bdm3_det_seed102-714203a25c | 0.211 | 0.106 | 0.058 | 0.189 |
| 20261104T023716Z-e1a_bdm3_det_seed103-f21e96d205 | 0.156 | 0.059 | 0.074 | 0.046 |
| 20261104T025951Z-e1a_bdm3_det_seed104-0a1b070df9 | 0.252 | 0.140 | 0.094 | 0.179 |
| 20261104T040839Z-e1a_bdm3_det_seed105-037eeaf732 | 0.114 | 0.069 | 0.117 | 0.116 |
