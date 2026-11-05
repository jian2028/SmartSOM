# e1g_balanced_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261104T130159Z-e1g_balanced_samp_seed101-0324bd9604 | 20 | 20 | 23.923 | 2685.550 | 94.261 | 29.500 | 0.481 | 0.744 | 0.110 |
| 20261104T130323Z-e1g_balanced_samp_seed102-830bd3ce0e | 20 | 18 | 24.645 | 2508.667 | 80.930 | 18.550 | 0.489 | 0.707 | 0.112 |
| 20261104T130602Z-e1g_balanced_samp_seed103-dfa8b03c09 | 20 | 18 | 24.171 | 2554.444 | 55.919 | 19.400 | 0.491 | 0.714 | 0.114 |
| 20261104T140142Z-e1g_balanced_samp_seed104-47691a6837 | 20 | 19 | 24.001 | 2624.684 | 51.646 | 18.250 | 0.478 | 0.716 | 0.102 |
| 20261104T140348Z-e1g_balanced_samp_seed105-53b334ec33 | 20 | 17 | 21.842 | 2794 | 123.556 | 31.700 | 0.461 | 0.695 | 0.117 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261104T130159Z-e1g_balanced_samp_seed101-0324bd9604 | 0.132 | 0.097 | 0.053 | 0.152 |
| 20261104T130323Z-e1g_balanced_samp_seed102-830bd3ce0e | 0.154 | 0.053 | 0.206 | 0.187 |
| 20261104T130602Z-e1g_balanced_samp_seed103-dfa8b03c09 | 0.162 | 0.063 | 0.140 | 0.066 |
| 20261104T140142Z-e1g_balanced_samp_seed104-47691a6837 | 0.052 | 0.106 | 0.214 | 0.417 |
| 20261104T140348Z-e1g_balanced_samp_seed105-53b334ec33 | 0.112 | 0.166 | 0.089 | 0.104 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed101 | 20 | 27 | 1.472 | 0.417 | 553.500 | 7 | 38.900 | 24.174 |
| seed102 | 20 | 23 | 1.033 | 0.455 | 419.048 | 2 | 14.778 | 25.943 |
| seed103 | 20 | 28 | 0.756 | 0.304 | 608.421 | 9 | 61.250 | 18.986 |
| seed104 | 20 | 21 | 0.849 | 0.143 | 445 | 1 | -66.722 | 2.080 |
| seed105 | 20 | 33 | 0.923 | 0.414 | 417.391 | 10 | 45.929 | 11.994 |
