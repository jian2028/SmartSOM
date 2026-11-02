# c1fix_balanced_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T004031Z-c1fix_balanced_det_seed101-12ce2b7264 | 20 | 18 | 24.982 | 2389.500 | 4.962 | 4.250 | 0.409 | 0.431 | 0.056 |
| 20261031T004031Z-c1fix_balanced_det_seed102-f2d3b8dad7 | 20 | 10 | 19.545 | 2286.100 | 57.746 | 5.700 | 0.313 | 0.460 | 0.160 |
| 20261031T004031Z-c1fix_balanced_det_seed103-d07a729073 | 20 | 18 | 24.390 | 2521.889 | 18.590 | 12.400 | 0.399 | 0.399 | 0.018 |
| 20261031T004031Z-c1fix_balanced_det_seed104-a240fc49e6 | 20 | 8 | 18.509 | 2406 | 6.434 | 5.450 | 0.325 | 0.486 | 0.144 |
| 20261031T004031Z-c1fix_balanced_det_seed105-47be3ee964 | 20 | 14 | 23.519 | 2361.857 | 15.885 | 5.450 | 0.375 | 0.388 | 0.038 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T004031Z-c1fix_balanced_det_seed101-12ce2b7264 | 0.710 | 0.753 | 0.705 | 0.238 |
| 20261031T004031Z-c1fix_balanced_det_seed102-f2d3b8dad7 | 0.189 | 0.177 | 0.582 | 0.224 |
| 20261031T004031Z-c1fix_balanced_det_seed103-d07a729073 | 0.743 | 0.659 | 0.702 | 0.752 |
| 20261031T004031Z-c1fix_balanced_det_seed104-a240fc49e6 | 0.565 | 0.185 | 0.171 | 0.549 |
| 20261031T004031Z-c1fix_balanced_det_seed105-47be3ee964 | 0.652 | 0.136 | 0.636 | 0.224 |
