# c1y_bdm3_samp (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T194615Z-c1y_bdm3_samp_seed101-5ed7c23f61 | 20 | 18 | 23.077 | 2688.944 | 93.245 | 26.800 | 0.470 | 0.712 | 0.113 |
| 20261031T194615Z-c1y_bdm3_samp_seed102-41502eb41c | 20 | 20 | 24.116 | 2664.500 | 121.237 | 31.250 | 0.495 | 0.744 | 0.123 |
| 20261031T194615Z-c1y_bdm3_samp_seed104-8e09db7f3c | 20 | 17 | 22.509 | 2701.412 | 124.580 | 32.850 | 0.458 | 0.722 | 0.114 |
| 20261031T194615Z-c1y_bdm3_samp_seed105-73dde15d64 | 20 | 19 | 22.331 | 2843.842 | 130.792 | 31.500 | 0.451 | 0.724 | 0.110 |
| 20261031T194616Z-c1y_bdm3_samp_seed103-e1542799f8 | 20 | 18 | 21.572 | 2892.222 | 160.255 | 35.200 | 0.440 | 0.704 | 0.109 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T194615Z-c1y_bdm3_samp_seed101-5ed7c23f61 | 0.261 | 0.067 | 0.164 | 0.089 |
| 20261031T194615Z-c1y_bdm3_samp_seed102-41502eb41c | 0.076 | 0.073 | 0.198 | 0.176 |
| 20261031T194615Z-c1y_bdm3_samp_seed104-8e09db7f3c | 0.171 | 0.296 | 0.162 | 0.162 |
| 20261031T194615Z-c1y_bdm3_samp_seed105-73dde15d64 | 0.358 | 0.068 | 0.254 | 0.142 |
| 20261031T194616Z-c1y_bdm3_samp_seed103-e1542799f8 | 0.317 | 0.169 | 0.374 | 0.600 |
