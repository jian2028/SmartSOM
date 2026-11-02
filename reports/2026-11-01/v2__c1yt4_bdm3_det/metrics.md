# c1yt4_bdm3_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261101T165755Z-c1yt4_bdm3_det_seed103-49442818c1 | 7 | 7 | 19.121 | 3354.286 | 340.583 | 43.429 | 0.458 | 0.650 | 0.123 |
| 20261101T165755Z-c1yt4_bdm3_det_seed105-995c8eeedc | 10 | 10 | 21.223 | 3049.800 | 183.897 | 33.200 | 0.483 | 0.661 | 0.103 |
| 20261101T165756Z-c1yt4_bdm3_det_seed102-5c65f8de6e | 11 | 10 | 24.061 | 2571.600 | 70.563 | 19.909 | 0.479 | 0.708 | 0.094 |
| 20261101T165757Z-c1yt4_bdm3_det_seed104-8e6e46cc5c | 10 | 9 | 24.157 | 2547 | 51.276 | 14.700 | 0.388 | 0.680 | 0.108 |
| 20261101T165758Z-c1yt4_bdm3_det_seed101-34b93c7ef4 | 10 | 10 | 21.847 | 2944.500 | 163.775 | 39.200 | 0.425 | 0.657 | 0.118 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261101T165755Z-c1yt4_bdm3_det_seed103-49442818c1 | 0.279 | 0.134 | 0.448 | 0.600 |
| 20261101T165755Z-c1yt4_bdm3_det_seed105-995c8eeedc | 0.301 | 0.167 | 0.434 | 0.237 |
| 20261101T165756Z-c1yt4_bdm3_det_seed102-5c65f8de6e | 0.179 | 0.159 | 0.297 | 0.260 |
| 20261101T165757Z-c1yt4_bdm3_det_seed104-8e6e46cc5c | 0.700 | 0.166 | 0.489 | 0.241 |
| 20261101T165758Z-c1yt4_bdm3_det_seed101-34b93c7ef4 | 0.631 | 0.691 | 0.462 | 0.345 |
