# 0021 — Travel Time Matrix and frozen Small H/V studies

Date: 2026-09-26
Status: Accepted implementation scope; engineering evidence is separate.

Supersedes ADR 0017's grid-only transportation requirement and ADR 0020's
per-tick Mover requirement **only for explicitly selected matrix scenarios**.
This does not restore the historical matrix runtime. ProductionSimulator remains
the sole physical kernel. Grid/v2/v3, historical recordings, inventory, quantity
reservations, automatic inspection and disposal retain their contracts.

## Transportation

Scenario selects grid or travel_time_matrix. A matrix file selects auto (BFS on
factory geometry) or manual (complete directed values, or an explicit default).
Auto can have sparse overrides. Zero manual times express teleportation; there
is no separate teleport backend. Point identities are factory port IDs and
initial:<agv_id>. Integers count physical ticks; null denotes unreachable. Missing
manual values without a declared default, unknown points and negative times fail
before execution. Diagonal times are zero. No triangle or symmetry requirement
is imposed on user values.

The core owns a nonpreemptive trip from boundary t to t+D. A zero trip arrives
before that boundary's service opportunities; each vehicle has at most one
Dispatcher proposal and one service per boundary. Loading/unloading still take
one tick; inspection/disposal remain physical operations. Matrix trips and queues
do not occupy road cells or require grid clearing. Ports admit one service at a
time, ordered by arrival among currently service-eligible vehicles, with private
identity-seeded queue ties. Full destinations remain selectable and check actual
capacity on unloading. There is **no destination-slot reservation**. Failed
arrival admission waits in an abstract queue; loaded vehicles can select another
legal endpoint after arrival. Transit vehicles have no Dispatcher or Mover action.

Matrix Mover is an explicit automatic_travel rule descriptor; models or movement
rules cannot be silently ignored. A central controller receives the remaining
conditional decisions. Dispatcher features, matching costs and rules use the
active travel provider. State, replay, snapshots and metrics retain trip timing.
Component metadata records physical contract and originating matrix digest;
changing matrix values within the matrix contract is explicit evaluation domain
change, while resume restores the exact saved matrix. Grid/matrix or rounding
contract mismatches require retraining, not silent reuse.

## H, V and bounded inputs

Machine processing_rate_multiplier defaults to one; quality modes retain their
existing identities and baseline time scales. Scenario processing_rounding
explicitly selects legacy half_up or new ceil. Small uses ceil after the complete
rate/mode calculation. G0/G1 freeze IDs, geometry and compatibility and preserve
per-operation/per-mode nominal rate and capacity-weighted yield. Report discrete
rounding losses separately; nominal invariants do not imply equal throughput.

Content recipes compile into explicit workload/v2 demands. V uses the current
model's DP edit cost divided by five, fixed reference-time bound, exact rational
min-cost coupling and compare-before-update history. A finite candidate set of
Job permutations is scored before learning; minimum, middle and maximum distinct
scores define relative Low/Mid/High. They are not universal numerical bins. Each
split's pool, slots, individual allowances, totals and initial history are frozen;
quality attempts never enter V. Initial history is not initial WIP. Train/val/test
pools are independently generated; identical standard specifications may occur.

New finite mode permits future arrivals and stops when all original demands have
qualified deliveries, otherwise truncates at the horizon. Existing static and
dynamic modes keep their semantics.

## Workflow and evidence

Composable studies compile 24 classified v3 run recipes and freeze detached
prepared snapshots, partners, data and source/dependency identities. Workers call
ordinary v3 train/evaluate/resume. Single-run manual execution can be adopted by
its scientific and implementation identity. Serial execution uses an exclusive
lock, retained failure state and explicit retry. Editing authoring files cannot
change frozen batch inputs. New source/dependencies require a new preparation.

Pilot training is 16384 ticks/entry; validation is five independently frozen cases
every 1024 ticks, with five held-out cases for final last-checkpoint evaluation.
Initial, rule and legal-random controls use the same cases. Initial network state
is saved before learning and evaluated only after training. Best retains whole-
composition validation provenance. This engineering/development workflow does
not authorize a research claim, milestone promotion, commit or formal launch.
