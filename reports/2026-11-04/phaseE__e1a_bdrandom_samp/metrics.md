# e1a_bdrandom_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T165034Z-e1a_bdrandom_samp_seed101-f18d1a9651 | 20 | 17 | 23.556 | 2562.118 | 58.676 | 21.200 | 0.474 | 0.684 | 0.112 |
| 20261104T170723Z-e1a_bdrandom_samp_seed102-7373ed153e | 20 | 17 | 22.537 | 2695.529 | 88.637 | 25.500 | 0.467 | 0.701 | 0.100 |
| 20261104T171202Z-e1a_bdrandom_samp_seed103-9214a93f52 | 20 | 18 | 23.974 | 2570.167 | 70.915 | 22.600 | 0.480 | 0.693 | 0.119 |
| 20261104T171216Z-e1a_bdrandom_samp_seed104-35b5352684 | 20 | 17 | 23.904 | 2526.588 | 52.747 | 19.600 | 0.479 | 0.686 | 0.109 |
| 20261104T171651Z-e1a_bdrandom_samp_seed105-ec635f2ab3 | 20 | 20 | 24.443 | 2625.600 | 92.252 | 26.200 | 0.493 | 0.723 | 0.117 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T165034Z-e1a_bdrandom_samp_seed101-f18d1a9651 | 0.075 | 0.113 | 0.098 | 0.109 |
| 20261104T170723Z-e1a_bdrandom_samp_seed102-7373ed153e | 0.293 | 0.069 | 0.090 | 0.140 |
| 20261104T171202Z-e1a_bdrandom_samp_seed103-9214a93f52 | 0.112 | 0.051 | 0.087 | 0.041 |
| 20261104T171216Z-e1a_bdrandom_samp_seed104-35b5352684 | 0.186 | 0.077 | 0.064 | 0.230 |
| 20261104T171651Z-e1a_bdrandom_samp_seed105-ec635f2ab3 | 0.120 | 0.092 | 0.080 | 0.076 |
