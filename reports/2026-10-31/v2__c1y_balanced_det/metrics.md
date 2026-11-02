# c1y_balanced_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T194615Z-c1y_balanced_det_seed101-dab70810e5 | 20 | 19 | 23.788 | 2667.158 | 99.474 | 20.550 | 0.460 | 0.673 | 0.113 |
| 20261031T194615Z-c1y_balanced_det_seed103-f604fbcdd2 | 20 | 19 | 21.636 | 2916 | 135.168 | 36.600 | 0.434 | 0.668 | 0.105 |
| 20261031T194615Z-c1y_balanced_det_seed104-82a021a1fe | 20 | 16 | 17.713 | 3508.812 | 586.868 | 47.500 | 0.413 | 0.658 | 0.141 |
| 20261031T194616Z-c1y_balanced_det_seed102-bb717154df | 20 | 16 | 22.570 | 2624.625 | 103.597 | 28.450 | 0.444 | 0.681 | 0.146 |
| 20261031T194616Z-c1y_balanced_det_seed105-8ba807b019 | 20 | 19 | 21.746 | 2926.368 | 211.728 | 36.450 | 0.346 | 0.711 | 0.163 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T194615Z-c1y_balanced_det_seed101-dab70810e5 | 0.245 | 0.215 | 0.616 | 0.176 |
| 20261031T194615Z-c1y_balanced_det_seed103-f604fbcdd2 | 0.679 | 0.200 | 0.680 | 0.706 |
| 20261031T194615Z-c1y_balanced_det_seed104-82a021a1fe | 0.294 | 0.597 | 0.571 | 0.284 |
| 20261031T194616Z-c1y_balanced_det_seed102-bb717154df | 0.164 | 0.136 | 0.268 | 0.155 |
| 20261031T194616Z-c1y_balanced_det_seed105-8ba807b019 | 0.641 | 0.119 | 0.578 | 0.246 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed101 | 20 | 27 | 0.940 | 0.409 | 555.714 | 6 | -22.929 | 22.365 |
| seed103 | 20 | 40 | 0.775 | 0.231 | 305.385 | 14 | 267.059 | 56.237 |
| seed104 | 20 | 40 | 0.860 | 0.325 | 377.931 | 11 | -100.900 | 1.247 |
| seed102 | 20 | 25 | 1.224 | 0.545 | 461.905 | 4 | -16.625 | 5.090 |
| seed105 | 20 | 35 | 0.796 | 0.303 | 242.593 | 8 | 122.250 | 17.851 |
