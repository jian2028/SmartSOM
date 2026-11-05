# e1a_bdm3_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T042446Z-e1a_bdm3_samp_seed101-c022c46bac | 20 | 18 | 24.168 | 2550.778 | 62.668 | 21.200 | 0.483 | 0.689 | 0.114 |
| 20261104T044434Z-e1a_bdm3_samp_seed102-be5a9b1c66 | 20 | 18 | 23.852 | 2583.611 | 56.538 | 21 | 0.493 | 0.721 | 0.098 |
| 20261104T044728Z-e1a_bdm3_samp_seed103-10a2d8ed85 | 20 | 20 | 25.169 | 2565.300 | 83.359 | 22.050 | 0.508 | 0.714 | 0.118 |
| 20261104T062405Z-e1a_bdm3_samp_seed104-a15fe33c00 | 20 | 20 | 25.266 | 2552.050 | 58.090 | 20.200 | 0.507 | 0.716 | 0.114 |
| 20261104T062609Z-e1a_bdm3_samp_seed105-b99936bdd7 | 20 | 19 | 24.067 | 2632.684 | 105.327 | 22.350 | 0.490 | 0.716 | 0.112 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T042446Z-e1a_bdm3_samp_seed101-c022c46bac | 0.061 | 0.079 | 0.119 | 0.091 |
| 20261104T044434Z-e1a_bdm3_samp_seed102-be5a9b1c66 | 0.283 | 0.082 | 0.078 | 0.157 |
| 20261104T044728Z-e1a_bdm3_samp_seed103-10a2d8ed85 | 0.113 | 0.045 | 0.072 | 0.045 |
| 20261104T062405Z-e1a_bdm3_samp_seed104-a15fe33c00 | 0.197 | 0.053 | 0.059 | 0.188 |
| 20261104T062609Z-e1a_bdm3_samp_seed105-b99936bdd7 | 0.131 | 0.065 | 0.066 | 0.080 |
