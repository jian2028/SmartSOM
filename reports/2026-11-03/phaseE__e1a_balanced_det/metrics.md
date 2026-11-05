# e1a_balanced_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261103T201216Z-e1a_balanced_det_seed101-8ebe6f3f72 | 20 | 14 | 22.574 | 2541.857 | 258.684 | 21.400 | 0.357 | 0.652 | 0.140 |
| 20261103T214419Z-e1a_balanced_det_seed102-3acd7e7b1f | 20 | 20 | 23.068 | 2780.950 | 105.893 | 28.450 | 0.548 | 0.727 | 0.078 |
| 20261103T214515Z-e1a_balanced_det_seed103-b0482b5e12 | 20 | 16 | 22.242 | 2684.812 | 116.375 | 26.300 | 0.439 | 0.682 | 0.144 |
| 20261103T214618Z-e1a_balanced_det_seed104-d5323a8871 | 20 | 19 | 25.074 | 2507 | 79.409 | 19.650 | 0.495 | 0.742 | 0.123 |
| 20261104T005953Z-e1a_balanced_det_seed105-8e28290ab0 | 20 | 19 | 24.601 | 2574.263 | 71.701 | 15.550 | 0.385 | 0.689 | 0.125 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261103T201216Z-e1a_balanced_det_seed101-8ebe6f3f72 | 0.130 | 0.104 | 0.104 | 0.078 |
| 20261103T214419Z-e1a_balanced_det_seed102-3acd7e7b1f | 0.242 | 0.128 | 0.152 | 0.231 |
| 20261103T214515Z-e1a_balanced_det_seed103-b0482b5e12 | 0.166 | 0.060 | 0.183 | 0.045 |
| 20261103T214618Z-e1a_balanced_det_seed104-d5323a8871 | 0.240 | 0.152 | 0.182 | 0.171 |
| 20261104T005953Z-e1a_balanced_det_seed105-8e28290ab0 | 0.127 | 0.075 | 0.177 | 0.126 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed101 | 20 | 29 | 0.971 | 0.286 | 243.478 | 6 | 22.357 | -110.174 |
| seed102 | 20 | 40 | 1.026 | 0.303 | 298.621 | 11 | 368.778 | 49.388 |
| seed103 | 20 | 30 | 1.017 | 0.375 | 426.250 | 6 | 96.364 | 18.205 |
| seed104 | 20 | 26 | 0.937 | 0.348 | 352.273 | 4 | 143.889 | 22.024 |
| seed105 | 20 | 26 | 0.751 | 0.286 | 395 | 2 | 158.588 | 30.257 |
