# c1_balanced_samp (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T155447Z-c1_balanced_samp_seed103-666e4b16b4 | 20 | 15 | 20.368 | 2674.867 | 52.479 | 20.200 | 0.430 | 0.494 | 0.083 |
| 20261030T155447Z-c1_balanced_samp_seed105-0774f2c24f | 20 | 11 | 17.673 | 2515.182 | 23.024 | 7.750 | 0.370 | 0.468 | 0.110 |
| 20261030T155448Z-c1_balanced_samp_seed102-a55c2989b9 | 20 | 10 | 16.978 | 2453.900 | 24.550 | 6.250 | 0.346 | 0.534 | 0.157 |
| 20261030T155448Z-c1_balanced_samp_seed104-ee82b0b1de | 20 | 11 | 18.245 | 2567.909 | 28.417 | 12.750 | 0.374 | 0.451 | 0.087 |
| 20261030T155449Z-c1_balanced_samp_seed101-5d929f5642 | 20 | 6 | 15.378 | 2514.833 | 32.346 | 12.700 | 0.337 | 0.397 | 0.069 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T155447Z-c1_balanced_samp_seed103-666e4b16b4 | 0.512 | 0.209 | 0.438 | 0.590 |
| 20261030T155447Z-c1_balanced_samp_seed105-0774f2c24f | 0.396 | 0.050 | 0.336 | 0.111 |
| 20261030T155448Z-c1_balanced_samp_seed102-a55c2989b9 | 0.137 | 0.100 | 0.337 | 0.207 |
| 20261030T155448Z-c1_balanced_samp_seed104-ee82b0b1de | 0.529 | 0.202 | 0.139 | 0.506 |
| 20261030T155449Z-c1_balanced_samp_seed101-5d929f5642 | 0.329 | 0.344 | 0.203 | 0.130 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed103 | 20 | 32 | 0.947 | 0.238 | 538.667 | 17 | 8.917 | 16.886 |
| seed105 | 20 | 34 | 0.792 | 0.167 | 532.857 | 20 | -50.200 | 14.540 |
| seed102 | 20 | 36 | 0.650 | 0.100 | 466.875 | 20 | 293.200 | 26.063 |
| seed104 | 20 | 34 | 0.770 | 0.211 | 475.833 | 22 | -0.400 | 13.760 |
| seed101 | 20 | 34 | 0.837 | 0.316 | 290.667 | 19 | 119.400 | 17.549 |
