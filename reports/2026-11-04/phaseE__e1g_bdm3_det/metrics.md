# e1g_bdm3_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T100336Z-e1g_bdm3_det_seed101-6dc2c060c2 | 20 | 18 | 24.630 | 2494.889 | 41.427 | 17.100 | 0.411 | 0.736 | 0.111 |
| 20261104T112204Z-e1g_bdm3_det_seed102-f38256cee3 | 20 | 20 | 25.201 | 2559.900 | 114.273 | 18.850 | 0.398 | 0.681 | 0.126 |
| 20261104T112300Z-e1g_bdm3_det_seed103-41bfac6f76 | 20 | 19 | 21.825 | 2903.947 | 153.288 | 31.250 | 0.519 | 0.666 | 0.104 |
| 20261104T112407Z-e1g_bdm3_det_seed104-b41ab5916f | 20 | 19 | 26.904 | 2331 | 15.581 | 6.850 | 0.432 | 0.664 | 0.082 |
| 20261104T112642Z-e1g_bdm3_det_seed105-a1cf957913 | 20 | 17 | 23.625 | 2563.059 | 29.038 | 11 | 0.371 | 0.638 | 0.092 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T100336Z-e1g_bdm3_det_seed101-6dc2c060c2 | 0.193 | 0.153 | 0.035 | 0.229 |
| 20261104T112204Z-e1g_bdm3_det_seed102-f38256cee3 | 0.142 | 0.148 | 0.293 | 0.261 |
| 20261104T112300Z-e1g_bdm3_det_seed103-41bfac6f76 | 0.194 | 0.120 | 0.165 | 0.111 |
| 20261104T112407Z-e1g_bdm3_det_seed104-b41ab5916f | 0.184 | 0.176 | 0.276 | 0.495 |
| 20261104T112642Z-e1g_bdm3_det_seed105-a1cf957913 | 0.173 | 0.461 | 0.202 | 0.347 |
