# c1_bdm3_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T155447Z-c1_bdm3_det_seed102-fae1342e23 | 20 | 7 | 16.049 | 2293.714 | 17.388 | 3.750 | 0.270 | 0.470 | 0.193 |
| 20261030T155447Z-c1_bdm3_det_seed105-5550058235 | 20 | 18 | 25.379 | 2408.056 | 47.500 | 11.600 | 0.407 | 0.466 | 0.065 |
| 20261030T155449Z-c1_bdm3_det_seed101-1742cf6009 | 20 | 15 | 23.460 | 2419.867 | 19.171 | 10.100 | 0.391 | 0.395 | 0.017 |
| 20261030T155449Z-c1_bdm3_det_seed103-90b08a4be9 | 20 | 15 | 22.174 | 2520.400 | 41.330 | 17.150 | 0.367 | 0.457 | 0.089 |
| 20261030T155449Z-c1_bdm3_det_seed104-2aa757d082 | 20 | 7 | 16.801 | 2418.714 | 9.859 | 6.300 | 0.295 | 0.411 | 0.101 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T155447Z-c1_bdm3_det_seed102-fae1342e23 | 0.147 | 0.134 | 0.385 | 0.174 |
| 20261030T155447Z-c1_bdm3_det_seed105-5550058235 | 0.715 | 0.182 | 0.500 | 0.237 |
| 20261030T155449Z-c1_bdm3_det_seed101-1742cf6009 | 0.693 | 0.704 | 0.486 | 0.235 |
| 20261030T155449Z-c1_bdm3_det_seed103-90b08a4be9 | 0.701 | 0.608 | 0.476 | 0.679 |
| 20261030T155449Z-c1_bdm3_det_seed104-2aa757d082 | 0.526 | 0.169 | 0.206 | 0.512 |
