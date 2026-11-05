# e1_balanced_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261103T144129Z-e1_balanced_det_seed101-4b6b4d3f8c | 20 | 19 | 21.529 | 2948.105 | 226.148 | 31.750 | 0.516 | 0.717 | 0.118 |
| 20261103T144129Z-e1_balanced_det_seed102-4fb7560539 | 20 | 18 | 24.756 | 2480.333 | 26.887 | 15.100 | 0.494 | 0.675 | 0.104 |
| 20261103T144129Z-e1_balanced_det_seed104-a4dc76143d | 20 | 18 | 23.442 | 2631.056 | 102.262 | 31.650 | 0.558 | 0.701 | 0.132 |
| 20261103T144130Z-e1_balanced_det_seed103-50dc4a2831 | 20 | 18 | 24.945 | 2463.722 | 57.007 | 17.500 | 0.410 | 0.700 | 0.138 |
| 20261103T160627Z-e1_balanced_det_seed105-dce4893c06 | 20 | 13 | 21.109 | 2599 | 131.318 | 22.300 | 0.363 | 0.650 | 0.137 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261103T144129Z-e1_balanced_det_seed101-4b6b4d3f8c | 0.178 | 0.193 | 0.099 | 0.184 |
| 20261103T144129Z-e1_balanced_det_seed102-4fb7560539 | 0.439 | 0.143 | 0.143 | 0.063 |
| 20261103T144129Z-e1_balanced_det_seed104-a4dc76143d | 0.152 | 0.081 | 0.098 | 0.257 |
| 20261103T144130Z-e1_balanced_det_seed103-50dc4a2831 | 0.050 | 0.091 | 0.076 | 0.126 |
| 20261103T160627Z-e1_balanced_det_seed105-dce4893c06 | 0.128 | 0.037 | 0.122 | 0.035 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed101 | 20 | 36 | 0.892 | 0.235 | 355.200 | 11 | 74.706 | 18.910 |
| seed102 | 20 | 24 | 0.804 | 0.300 | 488.500 | 4 | 83.294 | 14.163 |
| seed104 | 20 | 32 | 0.862 | 0.409 | 560.476 | 11 | 108.214 | 34.371 |
| seed103 | 20 | 23 | 0.864 | 0.250 | 502.632 | 4 | 49.471 | 11.785 |
| seed105 | 20 | 28 | 1.227 | 0.455 | 452.609 | 5 | -82.583 | 58.955 |
