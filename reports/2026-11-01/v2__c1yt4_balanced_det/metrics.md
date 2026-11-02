# c1yt4_balanced_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261101T165755Z-c1yt4_balanced_det_seed103-f860e16bf6 | 8 | 7 | 21.242 | 2897.429 | 234.068 | 38.875 | 0.506 | 0.681 | 0.121 |
| 20261101T165756Z-c1yt4_balanced_det_seed101-9162893148 | 10 | 10 | 22.450 | 2874.900 | 106.592 | 33.500 | 0.435 | 0.653 | 0.113 |
| 20261101T165756Z-c1yt4_balanced_det_seed105-dd06d412a0 | 10 | 10 | 22.752 | 2822.900 | 161.086 | 32.400 | 0.513 | 0.694 | 0.110 |
| 20261101T165757Z-c1yt4_balanced_det_seed104-d725f9db60 | 12 | 11 | 24.815 | 2494.636 | 28.778 | 9.167 | 0.395 | 0.663 | 0.097 |
| 20261101T165759Z-c1yt4_balanced_det_seed102-225f3c5a7a | 10 | 8 | 21.963 | 2715.500 | 86.896 | 19.900 | 0.437 | 0.660 | 0.093 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261101T165755Z-c1yt4_balanced_det_seed103-f860e16bf6 | 0.290 | 0.137 | 0.642 | 0.607 |
| 20261101T165756Z-c1yt4_balanced_det_seed101-9162893148 | 0.607 | 0.646 | 0.636 | 0.396 |
| 20261101T165756Z-c1yt4_balanced_det_seed105-dd06d412a0 | 0.326 | 0.198 | 0.666 | 0.192 |
| 20261101T165757Z-c1yt4_balanced_det_seed104-d725f9db60 | 0.700 | 0.148 | 0.642 | 0.235 |
| 20261101T165759Z-c1yt4_balanced_det_seed102-225f3c5a7a | 0.160 | 0.130 | 0.223 | 0.261 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed103 | 7 | 14 | 0.869 | 0.286 | 391.818 | 3 | 433.333 | 103.909 |
| seed101 | 10 | 19 | 0.828 | 0.312 | 355 | 5 | 69.600 | 57.183 |
| seed105 | 10 | 18 | 1.111 | 0.412 | 244.667 | 3 | 226.900 | 22.811 |
| seed104 | 10 | 11 | 0.603 | 0.100 | 547 | 1 | 99.750 | 24.169 |
| seed102 | 10 | 12 | 1.091 | 0.455 | 412.222 | 3 | -144.714 | -15.153 |
