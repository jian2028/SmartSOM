# c1y_bdm3_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T194615Z-c1y_bdm3_det_seed102-ca6c752781 | 20 | 18 | 23.704 | 2602.556 | 108.687 | 29.100 | 0.467 | 0.704 | 0.145 |
| 20261031T194615Z-c1y_bdm3_det_seed103-e3a7acbd2b | 20 | 18 | 19.831 | 3156.222 | 191.405 | 38.450 | 0.401 | 0.656 | 0.110 |
| 20261031T194615Z-c1y_bdm3_det_seed104-8ff9d0b45a | 20 | 11 | 17.134 | 3370.727 | 588.115 | 46.500 | 0.405 | 0.667 | 0.145 |
| 20261031T194615Z-c1y_bdm3_det_seed105-c7b37d99b8 | 20 | 17 | 20.393 | 3042.647 | 229.580 | 38.100 | 0.326 | 0.675 | 0.148 |
| 20261031T194616Z-c1y_bdm3_det_seed101-b9268c56e9 | 20 | 15 | 22.219 | 2608.467 | 121.838 | 25.600 | 0.433 | 0.661 | 0.120 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T194615Z-c1y_bdm3_det_seed102-ca6c752781 | 0.187 | 0.135 | 0.328 | 0.166 |
| 20261031T194615Z-c1y_bdm3_det_seed103-e3a7acbd2b | 0.667 | 0.182 | 0.479 | 0.659 |
| 20261031T194615Z-c1y_bdm3_det_seed104-8ff9d0b45a | 0.276 | 0.587 | 0.421 | 0.273 |
| 20261031T194615Z-c1y_bdm3_det_seed105-c7b37d99b8 | 0.602 | 0.138 | 0.408 | 0.239 |
| 20261031T194616Z-c1y_bdm3_det_seed101-b9268c56e9 | 0.244 | 0.201 | 0.397 | 0.181 |
