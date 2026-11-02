# c1_balanced_det (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261030T155446Z-c1_balanced_det_seed102-8d0dd81838 | 20 | 8 | 17.361 | 2243.750 | 9.856 | 2.200 | 0.282 | 0.454 | 0.172 |
| 20261030T155446Z-c1_balanced_det_seed104-014a0272fe | 20 | 9 | 17.925 | 2446 | 4.641 | 3.950 | 0.311 | 0.422 | 0.101 |
| 20261030T155446Z-c1_balanced_det_seed105-90db51b51d | 20 | 17 | 25.090 | 2376.941 | 37.087 | 8.150 | 0.397 | 0.461 | 0.077 |
| 20261030T155447Z-c1_balanced_det_seed103-17005f3a6f | 20 | 18 | 24.594 | 2479.944 | 63.065 | 12.750 | 0.403 | 0.441 | 0.057 |
| 20261030T155448Z-c1_balanced_det_seed101-8f0e10aa6e | 20 | 16 | 24.173 | 2398.938 | 6.189 | 5.100 | 0.397 | 0.400 | 0.024 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261030T155446Z-c1_balanced_det_seed102-8d0dd81838 | 0.161 | 0.147 | 0.520 | 0.197 |
| 20261030T155446Z-c1_balanced_det_seed104-014a0272fe | 0.548 | 0.170 | 0.176 | 0.544 |
| 20261030T155446Z-c1_balanced_det_seed105-90db51b51d | 0.692 | 0.196 | 0.684 | 0.224 |
| 20261030T155447Z-c1_balanced_det_seed103-17005f3a6f | 0.760 | 0.659 | 0.689 | 0.757 |
| 20261030T155448Z-c1_balanced_det_seed101-8f0e10aa6e | 0.701 | 0.721 | 0.683 | 0.232 |

## Paired breakdown metrics (vs the matching `bdm3` runs)

Adaptability = completions during each outage ÷ the same window without breakdowns (pass ≥ 0.95). Recovery = ticks after repair until 300-tick rolling throughput is ≥ 95 % of the control run.

| seed | pairs | outage_events | adaptability_mean | adaptability_pass_rate | recovery_ticks_mean | recovery_censored | makespan_increase | tardiness_increase |
|---|---|---|---|---|---|---|---|---|
| seed102 | 20 | 33 | 0.626 | 0.111 | 284.545 | 22 | 19.667 | 7.532 |
| seed104 | 20 | 33 | 0.833 | 0.235 | 559.333 | 18 | -14.500 | 5.218 |
| seed105 | 20 | 22 | 0.650 | 0.045 | 535.263 | 3 | 27.941 | 10.413 |
| seed103 | 20 | 25 | 0.611 | 0 | 493.684 | 6 | 47.214 | -21.736 |
| seed101 | 20 | 25 | 0.669 | 0.050 | 366.471 | 8 | 34.818 | 12.982 |
