# e1_bdm3_samp (phaseE)

Source runs under `runs/phaseE/`. Means over each run's evaluation episodes; `run_dir` names carry the training/eval seed.

| run_dir | episodes | completed | throughput_per_1000 | makespan | mean_tardiness | late_jobs | machine_util | agv_util | agv_loaded_waiting |
|---|---|---|---|---|---|---|---|---|---|
| 20261103T181058Z-e1_bdm3_samp_seed101-6b644cc998 | 20 | 20 | 24.460 | 2629.200 | 96.855 | 27.450 | 0.501 | 0.748 | 0.109 |
| 20261103T182016Z-e1_bdm3_samp_seed102-9a3c2a2c57 | 20 | 17 | 23.879 | 2532.176 | 65.607 | 21.300 | 0.484 | 0.695 | 0.111 |
| 20261103T195120Z-e1_bdm3_samp_seed103-1f5e2c6cc5 | 20 | 18 | 23.249 | 2664.278 | 103.262 | 30.350 | 0.473 | 0.710 | 0.124 |
| 20261103T200311Z-e1_bdm3_samp_seed104-0d282421ba | 20 | 19 | 24.487 | 2574.211 | 76.972 | 25.550 | 0.496 | 0.712 | 0.115 |
| 20261103T200516Z-e1_bdm3_samp_seed105-3fd8941151 | 20 | 20 | 25.324 | 2544.700 | 77.260 | 21.100 | 0.515 | 0.734 | 0.113 |

## Twin-machine utilisation gap (|u_a − u_b|)

| run_dir | 1/8 | 2/7 | 3/6 | 4/5 |
|---|---|---|---|---|
| 20261103T181058Z-e1_bdm3_samp_seed101-6b644cc998 | 0.153 | 0.095 | 0.081 | 0.148 |
| 20261103T182016Z-e1_bdm3_samp_seed102-9a3c2a2c57 | 0.356 | 0.095 | 0.054 | 0.060 |
| 20261103T195120Z-e1_bdm3_samp_seed103-1f5e2c6cc5 | 0.077 | 0.126 | 0.099 | 0.125 |
| 20261103T200311Z-e1_bdm3_samp_seed104-0d282421ba | 0.102 | 0.080 | 0.076 | 0.173 |
| 20261103T200516Z-e1_bdm3_samp_seed105-3fd8941151 | 0.086 | 0.056 | 0.052 | 0.052 |
