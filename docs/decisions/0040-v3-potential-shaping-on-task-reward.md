# 0040 — Potential-based shaping stacked on the V3 task reward

Date: 2026-10-10
Status: Proposed (draft for review; not implemented).

Supersedes only the ADR 0039 sentence "team/role reward transforms cannot stack
onto it", and only for registered potential-based shaping. Every other ADR 0039
contract is retained: the task scalar, its weights and references, gamma=1, the
finite objective window, privileged-only task accounting, and validation on task
return.

## Context

Centralized PPO with the Mover delegated to the `shortest_path_yield` rule learns
under the plain task reward (Batch02 small grid, 64 orders, seeds 101–105,
65,536 ticks): 29.5 ± 17.3 shipped per 4,096-tick episode, against 64 for the
rule baseline. Three seeds reach 40–44 and two stall at 8–13. The task signal is
sparse: between shipments the per-interval scalar is only the overdue-time term.
Earlier undelegated runs showed that a potential over AGV target distance changes
behavior monotonically with its weight (more pickups, fewer conflicts), but it was
tested only on legacy rewards because task rewards forbid stacking. Comparing task
reward with and without shaping is the planned next experiment.

## Decision

An experiment may set `training.reward.team` together with `training.reward.task`
when the team extension is a registered potential-based shaping extension. The
initial allow-list is `builtin.production_potential` and
`builtin.agv_distance_potential`. Its `gamma` parameter must equal the training
gamma, which the task already fixes to 1. Role reward transforms remain forbidden
with a task.

The learner scalar for each physical interval is
`task(t) + gamma * phi(s_{t+1}) - phi(s_t)`, followed by the existing positive
learner scale. `phi` reads only the public view the transition already carries:
AGV cells and targets, released jobs' public `remaining_steps`, and public
shipment identities. It never reads oracle quality, task totals or references.
`phi(s_{t+1}) = 0` exactly when the transition carries an end reason, so the task
window end is a shaping terminal. Optimizer update cuts carry no reason and are
not terminals, matching ADR 0039.

With gamma=1 the shaping sum over an episode is `-phi(s_0)`. It is fixed by the
case's initial state and identical for every policy on that case, so shaping
cannot change which policy maximizes task return.

## Accounting and identity

Task return in episode reports, validation and evaluation remains the ADR 0039
telescoped objective from committed core totals, never the shaped learner return.
Validation keeps `custom/return/max/ineligible`. A separate per-episode shaping
ledger (sum and initial potential) is reported beside it. The bound shaping
reference, parameters and code digest become part of the frozen task contract.
Models, run records and continuations carry them, and restore rejects a changed
shaping contract before mutable state loads, as for the task contract.

## Consequences

Required changes are limited to configuration validation (`RewardSpecV3` and the
v4 compile check), a shaping-ledger field in episode reports, and tests. Tests
cover: allowed and rejected combinations; gamma mismatch rejection; per-interval
learner scalar equal to task plus potential difference; episode shaping sum
`-phi(s_0)`; unchanged task return with and without shaping on a fixed trajectory;
and terminal handling at the window end versus update cuts. Default and legacy
recipes, and task experiments without shaping, are unchanged.

Shaping changes learning dynamics, not the objective. A better shaped run is
engineering evidence for one configuration, not a scheduling claim, and comparisons
must use the same seeds, cases and task return.
