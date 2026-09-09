# SmartSOM

SmartSOM is intended to become a modular, event-driven simulator and experiment
platform for dynamic flexible job shop scheduling problems. The long-term
direction includes composable dynamic events, production resources and
constraints, online policies, optimization solvers, evolutionary methods, and
optional single-agent and multi-agent learning adapters.

## Status

The first deterministic static core slice is implemented: separate immutable
factory/workload inputs, serial job chains, one or more processing modes per operation,
semantic dispatch, step/run/replay, stable completion ordering, in-memory trace,
actual makespan, and transition invariants.

The single-run configuration slice also supports strict five-file configuration,
seeded static JSP/FJSP generation or instance import, deterministic dispatch policies,
`validate`/`run`/`import-fjs` commands, and persisted run evidence including failures.

Static JSP also supports explicit waiting, exact schedule replay, SPT, and an
optional PyJobShop/CP-SAT adapter. The fixed ft06 reference and a real CP solution
both replay to makespan **55**. Static FJSP supports multiple modes, including
distinct modes on the same machine, and traditional `.fjs` import. The official
PyJobShop small example and Mk01 have fixed references and real adapter solutions
that replay to **6** and **40**. Online arrivals now add independent release/reveal
timing, filtered job observations, explicit event waiting and reproducible arrival
generation. Processing-time uncertainty adds independently materialized actual
durations while online policies continue to observe nominal durations.
Machine breakdown/repair adds independent fixed or seeded outage plans, paused
work that stays on its selected machine, and automatic continuation after repair.
It composes with arrivals, processing uncertainty and multiple modes, with exact
replay and independent fixed-break references. Fixed-matrix AGV transport now adds
complete input-to-output flow, multiple vehicles, explicit prebuffer rerouting,
and exact full-execution replay. Optional finite pre/post buffers add reservations,
loaded AGV waiting and completion blocking; without AGV, explicit instantaneous
transfers use the same logistics ownership. Configurable quality-speed modes add
shared/per-machine capability tables, independent quality draws and final output
inspection, composed with all existing modules. CP remains static-only.
Paired studies, bounded single-host process execution, recovery, live progress
and optional bounded debug logs are implemented. An optional shared AGV holding
buffer provides congestion fallback with capacity reservations and exact replay.
Frozen IDETC inputs and a 60-run SPT acceptance command are available; see the
[integration protocol and evidence boundaries](docs/validation/idetc-integration.md).
Its postcommit report, not the availability of the command, establishes acceptance.
The optional Gymnasium interface provides a shared, reveal-bound learning
projection and episode reward/limit handling without changing the kernel.
Optional RLlib PPO and SB3 Contrib MaskablePPO train through that interface;
verified checkpoints implement the ordinary online-policy contract for run/study.
The fixed-budget training and 15-run paired replay protocol is documented in
[learning acceptance](docs/validation/learning.md). Its postcommit report establishes
integrated acceptance; short training does not establish superiority over SPT.
CP runtime classification is deferred.
The [static core validation record](docs/validation/static-core.md) gives the
exact hand-calculated cases, boundaries, and verification commands.

## Frozen IDETC acceptance

The frozen IDETC study uses four cases, fixed SPT-M0/M1/M2 and five paired
replications (seed 101). Preview does not create run directories:

```sh
uv run smartsom plan configs/studies/idetc_spt.yaml
uv run python scripts/validate_idetc.py --workers 2
```

The acceptance command runs the ordinary batch engine, then verifies every run's
action and full schedule replay, observation hashes and paired quality outcomes.
Keep the terminal open or use a normal shell session manager; logs and attempts
remain under `artifacts/idetc/studies/`. It requires integrated main source; only
explicit `--development` results are marked as precommit diagnostics.

```sh
uv run smartsom batch --resume PATH_TO_STUDY
uv run python scripts/validate_idetc.py --study-dir PATH_TO_STUDY --workers 2
uv run python scripts/prepare_idetc.py --output-dir artifacts/idetc/export
```

The exporter verifies frozen byte hashes and refuses existing output directories.
No external IDETC checkout or learning dependencies are required.
Add `--retry-failed` to the resume command only when failed attempts should run again.

## Centralized learning

Install only the backends needed. The base package and SPT do not import them:

```sh
uv sync --locked --extra learning-rllib --extra learning-sb3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run --no-sync smartsom validate configs/runs/learning_rllib.yaml
uv run --no-sync smartsom train configs/runs/learning_rllib.yaml
uv run --no-sync smartsom train configs/runs/learning_sb3.yaml
```

These versioned training run files own seed 101, the 4096/1024-step budgets,
episode limits and output. The reusable scenario owns the case and enabled
modules; algorithm files own projection capacity, PPO parameters and provider.
`validate` performs preparation without simulation or an output directory.
Training freezes the base workload once and materializes generated disturbances
per episode; fixed imported inputs stay fixed. Training and paired evaluation use
separate seed recipes. Framework reset seeds never replace the scientific seeds.

Each successful training directory contains `checkpoint_algorithm.json`. Reference
that file from an ordinary `smartsom.run/v1` run or study algorithm entry to
evaluate it. `smartsom run` and `batch` never train implicitly. A checkpoint is
limited to the same base structure, modules and projection, with different random
realizations; using the algorithm preset on a new structure requires new training.
An optional evaluation budget sets `max_decisions` and `max_ticks` (defaults
1024/10000). It is separate from the CP solver-time budget.

```sh
uv run --no-sync python scripts/validate_learning.py \
  --rllib-training-dir PATH_TO_RLLIB_TRAINING \
  --sb3-training-dir PATH_TO_SB3_TRAINING \
  --output-dir artifacts/learning/acceptance --workers 2
```

This audits both episode ledgers, then runs the committed study template (root
202, five paired replications, two checkpoints and SPT) and verifies all actions,
full schedules, observation hashes and quality outcomes. The output directory
must be new. Training logs report samples, updates and episode outcomes; evaluation
retains normal run/study evidence. Generated models and runs are not committed.
Current backends use one local CPU environment and a two-layer 64-unit MLP;
RLlib is the mainline and SB3 the independent smoke path. Training-resume CLI and
cross-structure checkpoint generalization are not implemented.

The optional resource-agent interface is available with `uv sync --locked --extra
pettingzoo`: `SmartSOMParallelEnv` uses one agent per machine/enabled AGV, public
IDETC-style resource observations and current candidate tables. A deterministic
coordinator accepts or rejects joint proposals through the existing semantic step
API. NOOP is adapter-only; actual waiting and all physical records remain in the
kernel. See [resource acceptance](docs/validation/resource-marl.md) for scope,
independent timelines, source differences and the separate real-training gate.

`rllib.resource_ppo` trains a shared machine policy and a separate shared AGV
policy using that same interface. Use the existing case/algorithm/train/run/study
configuration split; no additional adapter file is required:

**Item 13 formal acceptance is pending.** The training/checkpoint/evaluation
extension is implemented. The original gate stalled in all five MARL evaluations. A user-authorized adjustment
scales only learner rewards by 0.0001; the same 4096 rounds and seeds now pass
all ten paired evaluations and replay (MARL mean 214.6, SPT 134.6). Environment
rewards remain in ticks, and NOOP, inputs and physics are unchanged. Both attempts
are retained; this is development evidence, not a new main-commit result. See the
[acceptance record](docs/validation/resource-marl.md) before using these commands.

```sh
uv sync --locked --extra learning-marl
uv run --no-sync smartsom validate configs/runs/learning_marl.yaml
uv run --no-sync smartsom train configs/runs/learning_marl.yaml
uv run --no-sync python scripts/validate_resource_learning.py \
  --training-dir PATH_TO_RESOURCE_TRAINING \
  --output-dir artifacts/resource-marl/acceptance --workers 2
```

The fixed micro has 4 jobs, 9 operations, 8 machines and 4 AGVs. Its resource
action spaces are 65 machine / 41 AGV indices, including always-available NOOP;
nonzero indices reference the current complete candidate table, not permanent
action slots. Training uses 4096 joint rounds (49152 agent decisions); this is
not the same sampling budget as 4096 centralized decisions. The acceptance script
audits training and runs 10 paired resource-PPO/SPT evaluations at study seed 202.
It requires completion and joint/action/schedule replay, with no superiority target.
Evaluation retains `joint_decisions.jsonl` in addition to existing physical evidence.
The exported `checkpoint_algorithm.json` can be reused in ordinary run/study
files for compatible cases; evaluation never starts training implicitly.

The acceptance command defaults to formal validation: a clean implementation
commit integrated into local `main`, with training, evaluation and audit all using
that same commit. A clean detached worktree at an integrated commit is supported.
Use `--development` for precommit/CI checks; these never establish formal acceptance.
The frozen recipe also checks all environment inputs and module switches, both
roles' optimizer records, and exactly one audited MARL/SPT pair per replication.
Linux validation is a separate Week3 follow-up; macOS results do not establish it.

## Python API

```python
from smartsom.dispatch import Dispatch
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.engine import Simulator, replay

factory = FactorySpec((Machine("M1"),))
operation = Operation("operation", (ProcessingMode("standard", "M1", 2),))
job = Job("job", (operation,))
workload = WorkloadInstance((Order("order", (job,)),))
simulator = Simulator(factory, workload)
decision = simulator.current_decision  # Immutable; reading does not advance time.
result = simulator.step(Dispatch("operation", "standard"))
assert result.makespan == 2
assert replay(factory, workload, result.actions) == result
```

`step()` returns the next `DecisionContext` or a terminal `SimulationResult`.
`run(policy)` repeatedly calls that same method; a policy needs only
`select_action(context) -> Dispatch | Transport | Transfer | WaitUntil | WaitNextEvent`. Rejected actions raise `InvalidActionError`
before changing state. A new simulator starts a new episode; completed instances
cannot be advanced again.

`WaitUntil(tick)` intentionally advances time, returning early at a completion
that permits a decision. It creates no lasting wait commitment. Complete
`ScheduledOperation` intervals can be passed to `replay_schedule(factory,
workload, schedule)`; it validates the whole schedule and checks every actual
interval after execution, including intentional idle time.

## Configured Runs

```bash
uv sync --locked
uv run smartsom validate configs/runs/generated.yaml
uv run smartsom run configs/runs/competition.yaml  # makespan 6
uv run smartsom run configs/runs/crossing.yaml     # makespan 5
uv run smartsom run configs/runs/generated.yaml   # seeded static JSP
```

Each run file references a scenario and algorithm; the scenario references a
factory and exactly one workload profile or instance. Relative paths belong to
the containing file, so the command works from another directory with an
absolute run-config path. Root seeds belong only in the run file.

```python
from smartsom.config import resolve_run
from smartsom.experiments import run_one

resolved = resolve_run("configs/runs/generated.yaml")  # No output directory.
result = run_one(resolved)  # Uses the resolved snapshot without rereading files.
print(result.simulation_result.makespan, result.run_dir)
```

Trace consumers can use `Simulator.trace_since(cursor)` to read an immutable
suffix without copying earlier records. The cursor is an integer from zero to
the current trace length; reading never advances or consumes the simulation.
The full `trace` and completed result remain available in memory.

Machine outages can be passed as `machine_events=MachineOutagePlan(...)` to
`Simulator`, action replay and schedule replay. Scenarios enable fixed input or
per-machine `exponential_uptime_v1` profiles; runs export independently reusable
`realized_machine_events.json`. Policies see current availability and pause state,
with actual net processing ticks disclosed only after completion. Repair times
and unfinished true remaining work stay private. See the
[machine-event contract and validation](docs/validation/machine-events.md).

```bash
uv run smartsom validate configs/runs/machine_events_generated.yaml
uv run smartsom run configs/runs/machine_events_fixed.yaml      # makespan 22
uv run smartsom run configs/runs/machine_events_generated.yaml  # makespan 26
uv run smartsom run configs/runs/machine_events_arrivals_event.yaml  # combined, 22
```

Runs save their resolved inputs, reusable `realized_instance.json`, manifest,
trace, metrics, summary, and progress under the configured output root. Failures
retain available evidence and a `failure.json`; CLI errors return nonzero.
See [configured-run validation and file contracts](docs/validation/configured-runs.md)
for the supported fields, generation rules, and exact acceptance cases.

Each scenario is a reusable case: factory, workload and enabled environment
behavior. Algorithm presets select stable provider IDs; there is no additional
case or adapter YAML. A `study.yaml` composes cases, algorithms, replications and optional module
disable variants without a separate authored run file for every pair. Standalone
root seeds remain in `run.yaml`; studies own their root and record child derivations.

Internally, input materialization, algorithm binding/construction and run evidence
have separate responsibilities. `run_one()` retains the shared step loop and
solver/replay lifecycle. See the [pre-item-7 refactoring record](docs/validation/pre-item7-refactor.md)
for compatibility checks, measured costs and deferred work. The CLI reports live stage/count progress. Study evidence defaults to observation
hashes; full snapshots are configurable and standalone defaults remain unchanged.
Debug is opt-in and bounded. See [study usage and recovery](docs/validation/studies.md)
and [ADR 0009](docs/decisions/0009-paired-studies-and-recovery.md).

```sh
uv run smartsom plan configs/studies/quality_compare.yaml
uv run smartsom batch configs/studies/quality_compare.yaml --workers 2
uv run smartsom batch --resume PATH_TO_STUDY
```

## Static JSP and FJSP algorithms

```bash
uv run smartsom run configs/runs/ft06_spt.yaml
uv sync --locked --extra cp
uv run --extra cp smartsom run configs/runs/ft06_cp.yaml
uv run --extra cp smartsom run configs/runs/pyjobshop_fjsp_cp.yaml  # 6
uv run --extra cp smartsom run configs/runs/mk01_cp.yaml           # 40
```

`builtin.spt` chooses the shortest current legal duration, breaking ties by
operation and mode IDs. It does not use future information or promise optimality.
`pyjobshop.cp_sat` requires explicit `full_static` visibility and uses the pinned
optional dependencies PyJobShop 0.0.9 and OR-Tools 9.12.4544, with one worker.
The run-owned `budget.solver_time_limit_seconds` defaults to 60 for CP and is
rejected for online providers. A missing extra produces an explicit error.

CP saves `solver_result.json` before replay. A complete feasible solution can
succeed without proof of optimality; the summary records `solver_status` and
`proven_optimal`. See the [item 3 acceptance record](docs/validation/static-jsp.md)
for the reference source snapshots, Python replay example, and base/CP checks.

## FJSP inputs

```bash
uv run smartsom run configs/runs/fjsp_fast.yaml            # selected fast mode: 4
uv run smartsom run configs/runs/fjsp_slow.yaml            # selected slow mode: 7
uv run smartsom run configs/runs/generated_fjsp_spt.yaml
uv run smartsom import-fjs data/reference/mk01/Mk01.fjs \
  --instance-id mk01 --output-dir artifacts/imported-mk01
```

The import command validates the whole input before creating a new directory,
then writes `factory.yaml` and `workload.json` for existing scenario/run files.
It refuses an existing output directory. The Python API is
`smartsom.workloads.import_fjs(path, instance_id="mk01")` and returns an immutable
`ImportedProblem` with factory, workload and import provenance.

[`static_fjsp.yaml`](configs/workloads/static_fjsp.yaml) samples candidate machines
independently per operation, one mode per candidate, then independently samples
each mode's fixed nominal duration. Operations may revisit machines. Only the
run's derived workload seed is consumed; exporting and reimporting an instance
keeps its content fixed when the run seed changes. The JSP generator retains its
original sampling recipe and golden output.

Handwritten instances can retain several modes on one machine, even with equal
durations and different IDs. Dispatch selects one mode for the entire operation.
SPT chooses the shortest currently legal pair; it does not wait for a busy faster
machine. See the [item 4 acceptance record](docs/validation/static-fjsp.md) for
the import grammar, generator recipe, source snapshots and replay checks.

## Online job arrivals

```bash
uv run smartsom validate configs/runs/online_arrivals_event.yaml
uv run smartsom run configs/runs/online_arrivals_dispatch.yaml  # makespan 6
uv run smartsom run configs/runs/online_arrivals_event.yaml     # makespan 6
uv run smartsom run configs/runs/generated_arrivals_event.yaml
```

`scenario.arrivals` selects a fixed timing table or `uniform_release_v1` profile.
Reveal exposes a full job and its release time; release permits processing.
Before reveal the job is absent from all decision fields. The default
`dispatch_available` trigger auto-advances when no dispatch is legal;
`arrival_event` also exposes arrival notifications with empty candidates.
SPT/first-feasible then return `WaitNextEvent()` without learning the next event
clock. Scripted policies must explicitly provide that action.

The Python engine and both replay APIs accept `arrivals=ArrivalPlan(...)` and
`decision_trigger=...`. All-zero plans preserve previous static traces exactly.
Runs additionally save `realized_events.jsonl` for reuse and `observations.jsonl`
for the snapshots actually delivered to policies. Workload and arrival digests
remain separate. Dynamic scenarios reject the static CP provider. See the
[item 5 contract and acceptance](docs/validation/online-arrivals.md) for the
schema, generator recipe, information boundary and independent hand reference.

## Processing-time uncertainty

```bash
uv run smartsom validate configs/runs/processing_generated.yaml
uv run smartsom run configs/runs/processing_fixed.yaml       # makespan 20
uv run smartsom run configs/runs/processing_generated.yaml   # makespan 23
uv run smartsom run configs/runs/processing_arrivals_event.yaml
```

`scenario.processing_time` selects a fixed realized table or an independent
`uniform_multiplier` profile (default 0.8–1.2). Workloads retain nominal durations;
the engine executes the chosen mode's actual duration. Candidates and SPT use
nominal values, and only completion reveals the selected mode's actual duration.
Unselected modes' realizations stay private.

`ProcessingTimePlan` is accepted through the `processing_times` keyword on
`Simulator`, `replay` and `replay_schedule`. Versioned semantic-ID draws and exact
half-up rounding preserve deterministic replay without runtime sampling.
`realized_processing_times.json` can be imported directly under a different seed
or provider. Unit multipliers preserve prior complete traces. Processing-time
uncertainty can compose with arrivals; it rejects the static CP provider.
See the [item 6 acceptance record](docs/validation/processing-times.md).

## Fixed-matrix AGV transport

```bash
uv run smartsom run configs/runs/transport_hand.yaml       # output delivery 15
uv run smartsom run configs/runs/transport_reroute.yaml    # reroute away from down M1: 7
uv run smartsom run configs/runs/transport_combined.yaml   # AGV + JA + MB + UPT
```

`factory.transport` defines nodes, machine locations, global input/output, AGVs
and a complete directed `travel_times` table. `scenario.transport: {kind:
fixed_matrix}` enables it. Times are strict nonnegative integers with zero diagonal;
empty and loaded travel share this matrix. Fixed logistics consumes no seed.

`Transport(agv_id, job_id, TransportDestination("machine", machine_id))` books a
ready job; `TransportDestination("output")` sends a fully processed job out.
Booking binds the job until delivery. Jobs may be rerouted between eligible
prebuffers, and processing mode is chosen only when `Dispatch` actually starts
processing. With buffers disabled, waiting areas are unlimited, including at busy/down machines.
SPT/first-feasible process first, otherwise choose the shortest empty-plus-loaded
trip. Their queue-rerouting filter requires a busy/down source and idle/up target.
Effective rules are recorded in algorithm parameters. Scripts can select other
legal transfers. Arrivals, outages and processing uncertainty compose freely.

Enable direct calls with `transport_enabled=True` on `Simulator`, `replay` or
`replay_schedule`. Full replay uses `result.execution_schedule`, an immutable
`ExecutionSchedule` of processing intervals and all sequenced transports.
Runs export `execution_schedule.json`; progress counts both completed operations
and jobs delivered to output. Makespan uses final output delivery, which may be
later than processing completion. The static CP provider rejects enabled transport.
See the [AGV contract and acceptance record](docs/validation/transport.md).

## Limited buffers

```bash
uv run smartsom validate configs/runs/buffers_direct_zero.yaml
uv run smartsom run configs/runs/buffers_direct_zero.yaml   # no AGV, makespan 6
uv run smartsom run configs/runs/buffers_vehicle_zero.yaml  # loaded wait, makespan 8
uv run smartsom run configs/runs/buffers_post_one.yaml      # blocking, makespan 6
uv run smartsom run configs/runs/buffers_combined.yaml      # AGV + JA + MB + UPT
```

Put per-machine capacities under `factory.buffers`, for example
`[{machine_id: M1, pre_capacity: 0, post_capacity: 1}]`, and enable
`scenario.buffers: {kind: limited}`. Zero means no waiting slot; null/missing means
infinite. Input/output and unspecified machines are infinite. This switch is
independent of transport and consumes no seed. Python entry points accept
`buffers_enabled=True`.

Positive prebuffer slots can be exclusively reserved by AGVs at booking. Full
or zero destinations may still be booked, but an unable-to-unload vehicle waits
loaded and stays occupied. Zero prebuffer unloads onto an empty, up machine
awaiting explicit dispatch. Completion into a full postbuffer leaves a completed
job holding the machine; post space or actual direct pickup frees it. A booked
job stays at its original source until pickup.

With AGV off, `Transfer(job_id, TransportDestination(...))` moves instantly only
when the destination can accept the job now. Both logistics modes require final
output movement. SPT/first-feasible prioritize processing and conservatively
filter AGV bookings; scripted policies may intentionally use loaded waiting.
Neither policy guarantees deadlock avoidance. Failure evidence distinguishes
physical deadlock from a policy with no safe action and no future event.

Finite/direct logistics export v2 execution schedules containing actual arrival
and unload times, direct transfers and explicit timed non-wait action order.
`replay_schedule` uses the same step loop and checks every record. Unlimited AGV
keeps v1 and its previous trace. Enabled buffers reject static CP even when all
capacities are infinite. See [the contract](docs/decisions/0007-finite-buffers-and-blocking.md)
and [acceptance/evidence](docs/validation/buffers.md). No automatic deadlock
recovery, shared buffer pools, swap moves or new dependencies are included.

## Quality-speed modes

```bash
uv run smartsom run configs/runs/quality_m0.yaml       # makespan 24, passing rate 1
uv run smartsom run configs/runs/quality_m1.yaml       # makespan 20, passing rate 0
uv run smartsom run configs/runs/quality_m2.yaml       # makespan 16, passing rate 0
uv run smartsom run configs/runs/quality_generated.yaml
uv run smartsom run configs/runs/quality_hidden.yaml
uv run smartsom run configs/runs/quality_machine.yaml
uv run smartsom run configs/runs/quality_combined.yaml
```

`factory.quality_speed` owns a default mode table and optional whole-table machine
replacements. `scenario.quality` enables generated operation draws or a fixed
`realized_quality.json`. A root-derived quality seed leaves existing seed domains
unchanged. Every base mode is combined with its machine's quality table; base
workload and UPT files remain reusable. Execution uses exact half-up rounding
(minimum one tick), applying the scale after the existing UPT realization.

Each operation evaluates one hidden draw at real completion. A defect is permanent,
but the job continues its full route and normal output. Policies learn job pass/fail
only at final output (last processing completion with logistics off). Set
`scenario.quality.probability_visibility: hidden` to hide configured error rates;
public is the default. Latent draws and unfinished quality outcomes always stay
private. Mode identity, machine, scale and effective nominal duration remain visible.

SPT/first-feasible accept `parameters.quality_mode: M0` for a fixed label, which
must exist on every candidate base mode; omitting it allows all choices. No
quality-aware objective is added. Enabled quality rejects the static CP provider.
`prepare_quality` supplies the immutable `quality=` input for Simulator and both
replay APIs. Successful results expose `result.quality.passing_rate`; files also
retain operation checks, final inspections, effective mode catalog and counts.
See [quality configuration and acceptance](docs/validation/quality-speed.md) and
[ADR 0008](docs/decisions/0008-quality-speed-and-final-inspection.md).

## Design Direction

- Keep a thin semantic simulation core and add behavior through composition.
- Represent decisions with stable entity-based actions rather than transient
  candidate positions.
- Keep simulator truth independent of algorithm and learning frameworks.
- Use one execution path for manual runs and batch experiments.
- Preserve local manifests, progress logs, semantic traces, metrics, summaries,
  and failures as the authoritative experiment record.
- Add optional dependencies only with the adapter that needs them.

The planned boundaries are documented in
[`docs/architecture.md`](docs/architecture.md). The staged implementation order
is documented in [`docs/roadmap.md`](docs/roadmap.md).

## Development

SmartSOM uses Python 3.12 and `uv`.

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
# Required when changing the CP adapter:
uv sync --locked --extra cp
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before creating a branch or commit.

## Shared holding buffer

```sh
uv run smartsom run configs/runs/holding_hand.yaml  # actual output makespan 11
```

Factory `holding_buffer` defines a buffer ID, transport node and capacity
(null/infinite or a nonnegative integer). Enable `scenario.holding_buffer:
{kind: shared}` alongside AGV transport. Ready unfinished jobs can move from a
postbuffer or completed-blocked machine into holding, then to a next eligible
machine. Pickup releases source capacity; reserved slots and loaded waiting use
the finite-buffer contract. No input/prebuffer parking or direct Transfer is added.
SPT uses holding only if no safe next machine can receive that job. Python APIs
accept `holding_buffer_enabled=True`; holding runs use v2 schedule replay. See
[the contract](docs/decisions/0010-shared-holding-buffer.md) and
[hand/combination acceptance](docs/validation/holding-buffer.md).
