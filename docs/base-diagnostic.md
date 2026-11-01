> Historical workflow record. Current commands and supported configuration are documented in [current-workflow.md](current-workflow.md) and ADR0058; reproduce older examples using their archived source.

# Reservation Base: initial diagnostic

New work uses the [native check/run interface](reservation-experiments.md). The
module commands below remain compatible and use the same runtime.

This opt-in implementation follows the approved 2026-10-25 SmartSOM Plan. It
executes the existing grid physics with a new reservation protocol. Old V3
nonexclusive intentions are not labeled as the new Base. Scientific comparison
results are unavailable until the recorded runs finish.

## Scene

| Parameter | Small | Target |
| --- | ---: | ---: |
| Machines / AGVs | 8 / 8 | 20 / 20 |
| Grid | 24 × 11 | 29 × 19 |
| Machines per A/B/C/D type | 2 | 5 |
| Pre / machine / post capacity | 4 / 1 / 4 | 4 / 1 / 4 |
| Total / initial orders | 128 / 16 | 320 / 40 |
| Mean later interarrival time | 36 | 14.4 |
| Initial drain candidate | 2,000 | 2,000 |

Factory files contain the exact approved coordinates: machine `(4+5c,3+4r)`,
pre/post at x−1/x+1, three interaction cells each, AGV `(4+5c,1+4r)`, and Input/
Output on boundary columns. Capacities do not enlarge footprints. Storage at
Input/Output is unlimited; road cells and ports remain exclusive. No inspection,
scrap, charging or extra wall entities are added.

Equal-probability product families rotate A→B→C→D, with an independently optional
repeat of the first operation (probability .5). Work is base 30/40/50/40 times
Uniform(.8,1.2). Ordinary/rush factors are Uniform(2,3)/Uniform(1.5,2.5), with
20% independent rush labels. Arrivals accumulate continuous exponential gaps,
then ceil each publication time. Due dates use ceil of factor × the shortest
legal standard-mode processing/loaded-transport chain, including load/unload and
mandatory same-machine continuation. Empty travel, waiting and failures are absent
from that nominal bound. Work is rounded only after the selected mode/rate conversion.

Modes slow/normal/fast use duration factors 1.2/1/.8 and per-operation defect
probabilities .01/.018/.03. Failure flags persist and only affect the one final
inspection. Main lambda is 1,000 tardiness ticks; 500/2,000 sensitivity is later.
The first run does not include random execution-time variation or AGV breakdown.

Normal/low/medium/high episodes are sampled independently with probability .25
each. Fault intervals are conditional truncated exponential with scale/bounds
600/[60,1800], 300/[30,900], 150/[15,450]. Repair is 60/[10,180] in every condition.
Every segment is ceiled, at least one tick. Machines start available; their
independent uptime clocks pause during repair. Extending calibration preserves
the same order/fault realization and does not clip repairs at old horizons.

## Four files and commands

`configs/base-diagnostic/{small,target}.yaml` reference an ordinary Factory v2
file, Base Workload v1, and Base Algorithm v1. Base Experiment v1 is deliberately
distinct from legacy Experiment v4. Its executable interface is:

```sh
uv run --no-sync python -m smartsom.experiments.base_diagnostic check configs/base-diagnostic/small.yaml
uv run --no-sync python -m smartsom.experiments.base_suite configs/base-diagnostic --output /absolute/new/run-directory --hours 24 --workers 3
```

The suite requires a clean source commit. Use the existing optional CPU Torch
environment; importing `smartsom` or executing rules does not import Torch. No
solver, Ray, GUI or external service is required by the new diagnostic runner.
The generic legacy CLI is unchanged and does not silently accept Base schemas.

The suite first calibrates both scales with two parallel workers. It covers both
rule and random across all four fault conditions, with separate calibration seeds.
At 2,000 drain ticks, still-progressing cases continue in 2,000-tick increments.
A 1,000-tick absence of production/transport handoffs is recorded as stalled and
does not justify extension. A configurable 10,000-tick candidate ceiling and the
wall budget fail calibration explicitly instead of extending forever. Every
condition needs a completed rule case before freezing; stalled random cases stay
recorded failures. An unresolved progressing case prevents learning launch.

After freezing per-scale limits, two short real-PPO pilots measure throughput.
Three seeds (101/102/103) then train concurrently, small before target, with fresh
initialization at each scale. Per-scale physical-tick budgets are identical across
seeds and use a conservative .45 fraction of pilot-estimated 12/6-hour capacity.
Training cuts use exactly the remaining physical budget. Wall-limited jobs retain
their last checkpoint and are explicitly incomplete. Total supervision is capped
at 24 hours, including calibration, pilots, training and evaluation.

Validation uses 32 shared cases and test uses 64, equally divided among conditions.
Each seed's fixed final PPO checkpoint is evaluated with argmax and sampling;
test sampling uses three action seeds. Rules run once per case; random uses three
test action seeds. Scenario randomness is separated from action randomness. The
same `SharedMover` code is used throughout. Up to eight evaluation processes use
one Torch thread each. No test-selected checkpoint or algorithm winner is declared.

## Controller and learning records

- Machine is mandatory job×mode; post/pre/Input own their distinct policy groups.
- Dispatcher has pickup/delivery heads and chooses stations, not ports or jobs.
- Pre/post reservation acceptance selects applicants or its independent STOP.
  Input fills quota without STOP. On-site service selects pairs/vehicles without STOP.
- Each candidate retains its semantic ID. Four separate typed sets encode local
  jobs, current applicants, existing commitments and allowed station summaries.
  Mean/max/count pooling uses zero vectors for empty sets. Width is 32; physical
  feature rows are 48-wide with fixed time/distance/count references 200/50/20,
  no clipping and no running observation normalization.
- Dispatcher receives per-site summaries, not remote job lists. Machine sees
  its processing candidates and local summaries/commitments. Buffer sees local
  jobs and direct downstream summaries. No learned peer/social-history module is
  added to Base.
- PPO records every decision, including deterministic choices (actor loss masked),
  with physical timestamp, entity action, old likelihood and current public critic
  input. Full-episode undiscounted return-to-go is shared across same-tick choices;
  the physical world ledger is never multiplied by their number. PPO clipping,
  advantage standardization and Huber critic loss are numerical learning choices.

The fixed Mover uses current occupancy and next-interval reservations with recursive
displacement. Loaded rush jobs have priority; ready service traffic precedes parked
traffic within a class, followed by due pressure and accumulated waiting. It can
select another port or real waiting cell, but cannot change a production destination.
It is not an optimal MAPF solver and makes no universal deadlock-freedom claim.
No production actor receives its occupancy map or a congestion ETA.

## Evidence and stopping

The suite root contains `manifest.json`, `budgets.json`, `progress.json`, worker
logs, per-worker frozen inputs, `episodes.jsonl`, `last.pt`, and evaluation
`cases.jsonl`/`summary.json`. `comparison.json` collects completed evaluation
summaries. A requested run directory must be new; existing artifacts are not reused.
Creating a `STOP` file in the suite root stops only its owned processes and prevents
new ones. A failed task retains `failure.json`; incomplete evaluation is not success.

Reports distinguish complete MT/joint cost from observed partial return and passing
rate. Full aggregate MT/joint cost is null if any matched case is incomplete;
the non-drained rate is always shown. Final quality failures are counted once.
`record_first_case: true` in the Experiment records the first evaluation case;
`episode(..., trace=stream)` also records actual moves/waits and semantic decisions.
`smartsom.trace.base.positions_at` reconstructs any recorded integer-tick AGV positions;
training does not record a detailed replay by default. These sparse records are
not legacy Studio snapshots and must not be opened as a legacy trace.

Still open: value-learning diagnosis and final algorithm selection; the paper's
primary sampling/argmax convention; social-learning mechanisms; lambda sensitivity;
formal integrated-main acceptance and a full Studio adapter for sparse Base traces.
The overnight run is an initial PPO Base/rule/random development comparison.
