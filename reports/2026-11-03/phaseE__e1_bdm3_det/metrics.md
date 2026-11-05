# e1_bdm3_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261103T161257Z-e1_bdm3_det_seed101-d3165b8736 | 20 | 18 | 20.559 | 3052.389 | 245.057 | 34.650 | 0.494 | 0.691 | 0.111 |
| 20261103T162955Z-e1_bdm3_det_seed102-2bd8611a12 | 20 | 19 | 24.529 | 2564.842 | 41.050 | 18.950 | 0.491 | 0.675 | 0.103 |
| 20261103T163602Z-e1_bdm3_det_seed103-a52d7cfc05 | 20 | 19 | 25.267 | 2492.737 | 68.792 | 20.350 | 0.416 | 0.704 | 0.133 |
| 20261103T180051Z-e1_bdm3_det_seed104-fea8527dc2 | 20 | 16 | 21.722 | 2753.938 | 136.632 | 35.100 | 0.519 | 0.667 | 0.122 |
| 20261103T180809Z-e1_bdm3_det_seed105-fe8e14930b | 20 | 17 | 22.802 | 2696.882 | 190.273 | 27.450 | 0.390 | 0.697 | 0.149 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261103T161257Z-e1_bdm3_det_seed101-d3165b8736 | 0.186 | 0.215 | 0.049 | 0.182 |
| 20261103T162955Z-e1_bdm3_det_seed102-2bd8611a12 | 0.446 | 0.130 | 0.074 | 0.063 |
| 20261103T163602Z-e1_bdm3_det_seed103-a52d7cfc05 | 0.053 | 0.092 | 0.031 | 0.104 |
| 20261103T180051Z-e1_bdm3_det_seed104-fea8527dc2 | 0.163 | 0.098 | 0.039 | 0.241 |
| 20261103T180809Z-e1_bdm3_det_seed105-fe8e14930b | 0.148 | 0.064 | 0.065 | 0.042 |
