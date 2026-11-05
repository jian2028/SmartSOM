# e1g_balanced_det (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T100336Z-e1g_balanced_det_seed101-5d429da6e4 | 20 | 19 | 25.039 | 2507.947 | 31.886 | 14.050 | 0.418 | 0.737 | 0.109 |
| 20261104T100336Z-e1g_balanced_det_seed102-2e7c948baf | 20 | 17 | 24.107 | 2515 | 105.813 | 15.050 | 0.379 | 0.654 | 0.128 |
| 20261104T100336Z-e1g_balanced_det_seed103-962de1cf69 | 20 | 17 | 22.391 | 2717.353 | 115.086 | 26.200 | 0.530 | 0.661 | 0.101 |
| 20261104T100336Z-e1g_balanced_det_seed104-f50dcfe8d0 | 20 | 18 | 26.670 | 2293.556 | 5.323 | 2.850 | 0.427 | 0.635 | 0.077 |
| 20261104T100336Z-e1g_balanced_det_seed105-c9820fca24 | 20 | 17 | 24.320 | 2467.765 | 11.922 | 6.200 | 0.380 | 0.659 | 0.096 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T100336Z-e1g_balanced_det_seed101-5d429da6e4 | 0.175 | 0.175 | 0.097 | 0.240 |
| 20261104T100336Z-e1g_balanced_det_seed102-2e7c948baf | 0.142 | 0.135 | 0.394 | 0.277 |
| 20261104T100336Z-e1g_balanced_det_seed103-962de1cf69 | 0.174 | 0.120 | 0.296 | 0.123 |
| 20261104T100336Z-e1g_balanced_det_seed104-f50dcfe8d0 | 0.189 | 0.174 | 0.406 | 0.509 |
| 20261104T100336Z-e1g_balanced_det_seed105-c9820fca24 | 0.177 | 0.475 | 0.291 | 0.367 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed101 | 20 | 23 | 0.815 | 0.286 | 522.778 | 5 | -10.353 | 9.541 |
| seed102 | 20 | 24 | 0.832 | 0.227 | 455.909 | 2 | 40.235 | 8.460 |
| seed103 | 20 | 36 | 0.723 | 0.308 | 361.379 | 7 | 188.375 | 38.202 |
| seed104 | 20 | 21 | 0.628 | 0 | 449.500 | 1 | 37.059 | 10.259 |
| seed105 | 20 | 27 | 0.647 | 0.048 | 468.571 | 6 | 110.643 | 17.116 |
