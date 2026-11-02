# c1yt4_bdm3_samp (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261101T165755Z-c1yt4_bdm3_samp_seed103-c0bfcb3d3b | 9 | 7 | 20.811 | 2885.571 | 139.191 | 29.778 | 0.426 | 0.693 | 0.119 |
| 20261101T165755Z-c1yt4_bdm3_samp_seed104-8a0e303825 | 10 | 10 | 22.618 | 2842.600 | 131.183 | 31.100 | 0.461 | 0.719 | 0.106 |
| 20261101T165755Z-c1yt4_bdm3_samp_seed105-d83079d1d2 | 11 | 10 | 23.953 | 2579.100 | 63.581 | 22.455 | 0.480 | 0.708 | 0.111 |
| 20261101T165757Z-c1yt4_bdm3_samp_seed102-1bc1bf45ad | 10 | 10 | 24.074 | 2667.400 | 128.395 | 28.200 | 0.487 | 0.739 | 0.110 |
| 20261101T165759Z-c1yt4_bdm3_samp_seed101-e19e53a348 | 9 | 9 | 23.298 | 2757.222 | 169.128 | 39.889 | 0.480 | 0.733 | 0.123 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261101T165755Z-c1yt4_bdm3_samp_seed103-c0bfcb3d3b | 0.185 | 0.109 | 0.254 | 0.466 |
| 20261101T165755Z-c1yt4_bdm3_samp_seed104-8a0e303825 | 0.523 | 0.109 | 0.275 | 0.224 |
| 20261101T165755Z-c1yt4_bdm3_samp_seed105-d83079d1d2 | 0.241 | 0.064 | 0.294 | 0.176 |
| 20261101T165757Z-c1yt4_bdm3_samp_seed102-1bc1bf45ad | 0.074 | 0.096 | 0.208 | 0.136 |
| 20261101T165759Z-c1yt4_bdm3_samp_seed101-e19e53a348 | 0.469 | 0.338 | 0.148 | 0.102 |
