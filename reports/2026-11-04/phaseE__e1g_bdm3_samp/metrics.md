# e1g_bdm3_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T113841Z-e1g_bdm3_samp_seed101-1b97397c61 | 20 | 20 | 23.618 | 2724.450 | 118.435 | 31.550 | 0.477 | 0.746 | 0.108 |
| 20261104T114520Z-e1g_bdm3_samp_seed102-1aafb40607 | 20 | 20 | 25.299 | 2554.450 | 106.873 | 21.500 | 0.500 | 0.727 | 0.109 |
| 20261104T123752Z-e1g_bdm3_samp_seed103-147f351add | 20 | 18 | 23.616 | 2621.278 | 74.906 | 24.850 | 0.482 | 0.716 | 0.115 |
| 20261104T124433Z-e1g_bdm3_samp_seed104-42bb0fc7f2 | 20 | 19 | 24.611 | 2550.947 | 53.726 | 19.900 | 0.490 | 0.728 | 0.094 |
| 20261104T124501Z-e1g_bdm3_samp_seed105-242a6f1762 | 20 | 17 | 21.782 | 2808.647 | 135.550 | 33.500 | 0.456 | 0.697 | 0.114 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T113841Z-e1g_bdm3_samp_seed101-1b97397c61 | 0.117 | 0.076 | 0.060 | 0.134 |
| 20261104T114520Z-e1g_bdm3_samp_seed102-1aafb40607 | 0.156 | 0.053 | 0.107 | 0.200 |
| 20261104T123752Z-e1g_bdm3_samp_seed103-147f351add | 0.173 | 0.076 | 0.089 | 0.069 |
| 20261104T124433Z-e1g_bdm3_samp_seed104-42bb0fc7f2 | 0.079 | 0.122 | 0.130 | 0.399 |
| 20261104T124501Z-e1g_bdm3_samp_seed105-242a6f1762 | 0.115 | 0.132 | 0.124 | 0.089 |
