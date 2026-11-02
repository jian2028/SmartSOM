# c1yield_balanced_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T025356Z-c1yield_balanced_det_seed101-f450435898 | 20 | 19 | 24.183 | 2606.895 | 44.554 | 19.550 | 0.395 | 0.661 | 0.128 |
| 20261031T025356Z-c1yield_balanced_det_seed104-2ab965dcc2 | 20 | 16 | 21.842 | 2736.875 | 210.259 | 34.800 | 0.371 | 0.687 | 0.163 |
| 20261031T025357Z-c1yield_balanced_det_seed102-e0b031a65d | 20 | 16 | 22.049 | 2701.938 | 143.250 | 27.850 | 0.342 | 0.655 | 0.159 |
| 20261031T025357Z-c1yield_balanced_det_seed103-1d02a28e2b | 20 | 17 | 23.067 | 2616.706 | 59.840 | 24.700 | 0.376 | 0.643 | 0.130 |
| 20261031T025357Z-c1yield_balanced_det_seed105-71c31b84f7 | 20 | 19 | 25.100 | 2500.737 | 38.571 | 15.350 | 0.399 | 0.666 | 0.126 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T025356Z-c1yield_balanced_det_seed101-f450435898 | 0.653 | 0.690 | 0.645 | 0.262 |
| 20261031T025356Z-c1yield_balanced_det_seed104-2ab965dcc2 | 0.504 | 0.220 | 0.183 | 0.634 |
| 20261031T025357Z-c1yield_balanced_det_seed102-e0b031a65d | 0.127 | 0.213 | 0.629 | 0.269 |
| 20261031T025357Z-c1yield_balanced_det_seed103-1d02a28e2b | 0.730 | 0.586 | 0.643 | 0.727 |
| 20261031T025357Z-c1yield_balanced_det_seed105-71c31b84f7 | 0.676 | 0.161 | 0.653 | 0.231 |
