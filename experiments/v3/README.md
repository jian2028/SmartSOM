# V3 completion-cost experiments

Derived from the integrated V2 campaign at `e4b1193`; V2 files are not modified.
Small, Medium and Large each contain 17 recipes: Base/ISL × shared/independent ×
seeds 101–103, Morris × three seeds, Justin and Random. **This campaign runs only
Small on Mac and Medium on BudgetVM, excluding Morris. Large is authored only.**
W&B remains online (`SmartSOM`, groups `v3-small`, `v3-medium`, `v3-large`).

## Controlled changes

- Learning/rule-candidate pickup uses NPF, HPOF, EDSF, LPCL. The V2 rule IDs and
  observation feature slots remain stable so the existing networks are unchanged;
  removed rules are not available actions. Justin and centralized Morris preserve
  their own existing policy/action definitions as comparison methods.
- HPOF orders by descending POST occupancy ratio, then descending resident PRE
  count plus the machine's resident job (0/1), then semantic candidate identity.
  No POST supply means HPOF is unavailable. Distance is not an added tie-break.
- Team reward is `-(dt + overdue_delta/N + 1000*failed_delta/N)/1000`.
  N is the fixed original-order count, not the delivered count. Failure is counted
  once at shipment. `gamma=1`, learner reward scale=1; no extra 0.001 scaling.
  Idle time/utilization are diagnostics, not reward terms. This baseline weighting
  is a design choice, not a validated optimum or a hard quality constraint.
- `episode_end: all_delivered` stops at the final committed Output shipment,
  including previously scheduled future arrivals. Failed shipments still close
  their original demand and incur quality cost; this adds no rework/replacement.
  `drain_ticks` is now the expansion chunk for these recipes, not the episode cap.
  The official runtime extends the same seeded world before a horizon boundary;
  existing arrivals and outages are verified unchanged. New outages continue from
  the same seeded process. A shorter safety/window objective is not substituted.
- Best/last retention and periodic evaluation schedule are inherited. Best uses
  `periodic_evaluation.cost_metric: learning_cost`, the negative accumulated team
  reward, censored unless all demands finish. The historical `joint_cost` remains
  the V2 overdue/quality ledger, **not V3's selection objective**. Final testing
  still uses the prescribed episode 1000 rather than selecting on test data.

Orders, operation-count distributions, release/due-date generation, maps,
resources, disturbances, non-pickup rules, observations, networks, seeds and
other training settings inherit the corresponding V2 recipe. Test world IDs and
initial world hashes remain V2's for paired comparison. An extended run records
its expansion boundaries; its final horizon is not claimed to be that initial
hash. The full recorded scenario includes its realized outage extension.

## Official entry points

Generate: `uv run python scripts/build_v3_experiments.py`.
Validate all maps without launching: `uv run smartsom check experiments/v3`.
Use `smartsom run`, never launch the whole V3 root: choose one map and explicitly
select Base/ISL shared/independent, Justin and Random IDs, excluding Morris.
List seed 101 entries first; choose runtime concurrency from current host capacity.
Example (limited selection; add other desired IDs explicitly):

```sh
uv run smartsom run experiments/v3/small \
  --id base-shared-101 --id base-independent-101 \
  --id isl-shared-101 --id isl-independent-101 --id justin-101 --id random-101 \
  --output-root runs/v3-small --max-concurrent 2 --background
```

Use official `status`, `monitor`, `configure`, `pause`, `resume`, `stop`, `add`
commands for lifecycle changes. User-authorized branch runs must record the exact
committed source and frozen inputs; never run from this editable development tree
while modifying it. Do not migrate old V2 checkpoints into the changed action set.

## Evidence boundaries

Complete reports expose makespan, total/mean tardiness, failed count/rate,
learning_return and learning_cost. Aborted/interrupted runs do not receive a fake
completed makespan or terminal success; retain their status and observed costs.
Full completion does not guarantee good quality, convergence or superiority.
