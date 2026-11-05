# E1a centralized PPO — status summary (2026-11-05)

Engineering evaluation evidence from local runs; not a formal experiment
(runs were made from the local SmartSOM-phaseE working copy, not an integrated
`main` commit).

## E1a spec

- Central PPO dispatcher (joint job → machine → AGV); AGV motion by the
  `shortest_path_yield` rule. Config: `configs/algorithms/central01_v2_e1a.yaml`,
  run: `configs/runs/central01/v2/e1a_train.yaml`.
- Reward: shipment 0.5, passing 0.1, tardiness 2.0, starvation 1.0 (the only
  difference from E1).
- Network: actor/critic 64×64 tanh, dispatch supply mask + commit, target
  machine features, joint destination. lr 1e-4, entropy 0.03, γ = 1.0.
- Training: factory_007 (19×9 grid), balanced workload, no breakdowns,
  262,144 ticks (1024 updates) per seed, seeds 101–105.

## Results (sampling inference; 5 seeds × the same 20 eval episodes)

Justin+yield = rule dispatcher + yield mover; its seeds 102–105 reproduce seed
101 exactly, so it is effectively one deterministic result on the same 20
episodes. gap = `twin_gap` from `scripts/morris/v2_metrics.py`.

### No breakdowns (balanced)

| Model | Completed | Throughput/1000 | Makespan | Mean tardiness | Late jobs | Machine util | gap |
|---|---|---|---|---|---|---|---|
| **E1a** | 90% | 24.53 | 2516 | 55.0 | 16.9 | 0.495 | 0.11 |
| E1 | 94% | 24.37 | 2573 | 69.2 | 22.0 | 0.494 | 0.11 |
| E1g | 92% | 23.72 | 2633 | 81.3 | 23.5 | – | 0.14 |
| C1y | 92% | 23.47 | 2661 | 94.7 | 27.8 | 0.475 | 0.24 |
| Justin+yield | 95% | 24.31 | 2591 | 46.5 | 13.5 | 0.522 | 0.16 |

### Fixed breakdowns (bdm3)

| Model | Completed | Throughput/1000 | Makespan | Mean tardiness | Late jobs | Machine util | gap |
|---|---|---|---|---|---|---|---|
| **E1a** | 95% | 24.50 | 2577 | 73.2 | 21.4 | 0.496 | 0.10 |
| E1 | 94% | 24.28 | 2589 | 84.0 | 25.1 | – | – |
| E1g | 94% | 23.79 | 2652 | 97.9 | 26.3 | – | – |
| C1y | 92% | 22.72 | 2758 | 126.0 | 31.5 | – | 0.21 |
| Justin+yield | 95% | 23.76 | 2665 | 66.8 | 18.1 | 0.511 | 0.14 |

### Random breakdowns (bd_random)

| Model | Completed | Throughput/1000 | Makespan | Mean tardiness | Late jobs | Machine util | gap |
|---|---|---|---|---|---|---|---|
| **E1a** | 89% | 23.68 | 2596 | 72.6 | 23.0 | 0.479 | 0.11 |
| E1 | 95% | 23.84 | 2648 | 92.7 | 27.1 | – | – |
| E1g | 94% | 23.57 | 2667 | 107.5 | 28.4 | – | – |
| C1y | 90% | 21.99 | 2825 | 148.2 | 33.9 | 0.446 | 0.22 |
| Justin+yield | 100% | 23.83 | 2704 | 89.6 | 23.2 | 0.513 | 0.14 |

### Adaptability / robustness (deterministic, balanced vs bdm3 paired)

| Model | Adaptability | Pass rate | Recovery ticks | Makespan increase | Tardiness increase |
|---|---|---|---|---|---|
| **E1a** | 0.940 | 32% | 343 | 158 | 1.9 |
| E1 | 0.930 | 33% | 472 | 46.6 | 27.6 |
| C1y | 0.919 | 36% | 389 | 49.8 | 20.6 |
| E1g | 0.729 | 17% | 452 | 73.2 | 16.7 |
| Justin+yield | 0.714 | 18% | 439 | 75.8 | 20.3 |

## Observations

- E1a is the best RL variant in every condition; vs C1y, mean tardiness is
  ~40–50% lower.
- Justin+yield has lower tardiness without breakdowns and under bdm3; under
  bd_random E1a has lower tardiness (72.6 vs 89.6) and makespan (2596 vs 2704),
  and higher adaptability (0.94 vs 0.71).
- E1g (bd_random + per-episode outage resampling in training) did not help.
- Known limitation: E1a completion is 89–95%. Every truncated episode is a
  stall with ~2 jobs left; a recorded probe shows a head-on livelock of two
  AGVs on the single-lane top corridor (mover level, `shortest_path_yield`).
  Three local mover fixes made congestion worse and were not adopted; the
  stall rate is accepted as a mover limitation. Mean tardiness counts shipped
  jobs only; the lower-bound-corrected value (E1a 60.5, E1 72.4, E1g 110.2 on
  the final eval) keeps the same ranking.

Per-run tables: see the `phaseE__e1a_*`, `phaseE__e1_*`, `phaseE__e1g_*`,
`phaseE__c1y_bdrandom_samp` and `*rules_justin_yield*` reports listed in
`reports/README.md`.
