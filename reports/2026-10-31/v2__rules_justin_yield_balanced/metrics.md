# rules_justin_yield_balanced (v2)

Source runs under `runs/v2/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261031T085732Z-rules_justin_yield_balanced-ba48ef88cf | 20 | 19 | 24.308 | 2591.474 | 46.481 | 13.500 | 0.522 | 0.610 | 0.097 |
| 20261105T005710Z-rules_justin_yield_balanced_seed104-205e77a5ba | 20 | 19 | 24.308 | 2591.474 | 46.481 | 13.500 | 0.522 | 0.610 | 0.097 |
| 20261105T005711Z-rules_justin_yield_balanced_seed102-e37beec7fd | 20 | 19 | 24.308 | 2591.474 | 46.481 | 13.500 | 0.522 | 0.610 | 0.097 |
| 20261105T005711Z-rules_justin_yield_balanced_seed105-ee0c6ae0b5 | 20 | 19 | 24.308 | 2591.474 | 46.481 | 13.500 | 0.522 | 0.610 | 0.097 |
| 20261105T005712Z-rules_justin_yield_balanced_seed103-7ef433d431 | 20 | 19 | 24.308 | 2591.474 | 46.481 | 13.500 | 0.522 | 0.610 | 0.097 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261031T085732Z-rules_justin_yield_balanced-ba48ef88cf | 0.073 | 0.254 | 0.268 | 0.056 |
| 20261105T005710Z-rules_justin_yield_balanced_seed104-205e77a5ba | 0.073 | 0.254 | 0.268 | 0.056 |
| 20261105T005711Z-rules_justin_yield_balanced_seed102-e37beec7fd | 0.073 | 0.254 | 0.268 | 0.056 |
| 20261105T005711Z-rules_justin_yield_balanced_seed105-ee0c6ae0b5 | 0.073 | 0.254 | 0.268 | 0.056 |
| 20261105T005712Z-rules_justin_yield_balanced_seed103-7ef433d431 | 0.073 | 0.254 | 0.268 | 0.056 |
