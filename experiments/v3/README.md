# V3 Rule-calibrated drain experiments

Derived from the integrated V2 campaign at `e4b1193`. Small, Medium and Large each
contain 17 recipes: Base/ISL × shared/independent × seeds101–103, Central × three
seeds, Rule and Random. Run Small on Mac and Medium on BudgetVM. **Central and
Large are authored only and are excluded from this campaign's launch selection.**
W&B stays online in project SmartSOM, with v3-small/medium/large groups.

## Scientific settings

- Pickup candidates: NPF, HPOF, EDSF, LPCL. V2 semantic rule IDs and actor feature
  dimensions remain fixed. HPOF ranks POST occupancy ratio, then resident PRE
  count plus the machine's resident job, then semantic identity. Central/Rule
  preserve their own existing policy/action definitions.
- Team reward: `-(dt + overdue_delta/N + 1000*failed_delta/N)/1000`, fixed original
  demand count N. A failed shipment is counted once and closes its demand.
  At an unfinished drain deadline, subtract `10 + 10*unfinished/N` once,
  outside the /1000 scaling. No completion bonus. These are initial coefficients.
  Gamma=1, learner reward scale=1. Idle time/utilization remain diagnostics.
  General completion-reward parameters `truncation_fixed_penalty` and
  `truncation_unfinished_weight` default to zero; V3 explicitly sets both to 10.
  Native `drain_deadline_terminal: true` retains DQN's final decision return
  (including this penalty) without next-state bootstrap, and sets PPO's boundary
  value to zero. Budget/wall interruptions retain their previous treatment.
  These settings do not change physical completion status or network structure.
- `episode_end: drain_deadline` uses the existing official runtime: finish at the
  last Output shipment, or truncate at last arrival + drain_ticks. This replaces
  the initial completion-only V3 setting that could wait forever after a deadlock.
  No additional admission constraint or emergency capacity. See ADR0084 for costs.
- For incomplete cases, makespan, final tardiness, learning_cost and completion
  tail remain null. Partial costs/coverage remain observable; a cutoff is not a
  successful completion. Best selection still uses learning_cost with existing
  incomplete-case eligibility. Final test data never selects a checkpoint.

## Rule calibration and result files

Official Rule evaluation now writes `completion_tail_ticks` in each `cases.jsonl`
row, only when completed. Each method/condition (including all) in `summary.json`
contains `completion_tail`: completed/measured/noncompleted/missing counts,
min/median/max ticks, complete_coverage and suggested_drain_ticks. The latter is
`ceil(1.5 * max completed tail)`, not 1.5 times total makespan. The field is also
available in research summaries and W&B.

Small and Medium use 32 calibration cases each (eight paired seeds in domain71000 ×
four conditions). Calibration seeds are independent of final testing. Failed
or stalled cases remain reported and do not contribute a fabricated completion
time; a successful-case maximum does not guarantee every case can finish.

Each `experiment.yaml` directly authors its own integer `drain_ticks`. The
selected Rule value is copied into these files once; there is no shared config,
automatic result lookup or synchronization. Each recipe can later be adjusted
independently. The generator preserves each existing recipe's authored limit and
re-pins its world hashes. Large remains uncalibrated and is not run. Calibration
uses an initial10000tick observation window and records exact inputs/source.

Scenario, job-operation distributions, maps, disturbances, networks, training
seeds and other V2 settings stay inherited. Test world seeds and action repeats
are preserved; world hashes are regenerated for the chosen horizon/outage suffix.

## Official entry points

Generate: `uv run python scripts/build_v3_experiments.py`.
Check: `uv run smartsom check experiments/v3`.
Launch only one map and explicitly select desired non-Central IDs, seed101 first:

```sh
uv run smartsom run experiments/v3/small \
  --id base-shared-101 --id isl-shared-101 --id rule-101 \
  --id base-independent-101 --id isl-independent-101 --id random-101 \
  --output-root runs/v3-small --max-concurrent 2 --background
```

Use official status/monitor/configure/pause/resume/stop/add controls. Preserve old
V3 run directories and complete checkpoints. A change to drain or scientific
inputs requires a newly frozen run; do not overwrite existing continuations.

## Recorded calibration (2026-11-09)

| Map | Completed / cases | Maximum completed tail | Authored drain_ticks |
| --- | --- | --- | --- |
| Small | 30 / 32 | 6983 | 10475 |
| Medium | 32 / 32 | 6914 | 10371 |

Small has two censored cases; its value covers observed successful tails only.
Large retains 10000 without calibration. No truncation penalty is added.
