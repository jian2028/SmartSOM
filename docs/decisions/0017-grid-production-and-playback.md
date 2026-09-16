# 0017 — One grid production contract with live and recorded views

> Checkpoint boundary: the headless grid runtime, training, recording and audit
> are implemented here. Live rendering and the graphical playback controls
> described below are the accepted design for the following viewer checkpoint.

Date: 2026-09-14

Status: Execution contract accepted and implemented. Development verification
and remaining native interaction limits are listed in
[implementation status](../production-runtime.md). Local commits do not promote
historical research acceptance results.

## Decision

Use Studio's grid factory definition as the factory input to the production core.
Do not offer two runtime generations or silently execute the earlier matrix model
when a grid feature is unsupported. The version in a persisted schema identifies
its contract; it is not a user-facing choice between simulators. Workload steps
refer to operation-type IDs, with machine eligibility derived from capabilities.
Studio's editor preferences remain outside physical semantics, in the same file.

This supersedes ADR 0014/0016's deferral of runtime integration, ADR 0006's fixed
matrix/whole-trip transport actions, ADR 0007's missing-capacity-as-unlimited
convention, ADR 0008's final-only inspection/no-replacement lifecycle, and ADR
0011/0012's earlier projection, action and episode-reward contracts for new grid
execution. Current reward accounting is specified in the runtime contract; it is
not the historical negative-makespan return.
Their historical records and results remain tied to their original commits.
ADR 0015's static editing/document lifecycle remains in force. The core still owns
physics, algorithms own decisions, and observation projections hide latent truth.

The frozen `PreparedExperiment`/`ProductionRecipe` boundary replaces ADR 0002's
matrix `ResolvedRun` payload for new execution. Both public `run` and study workers
call the same `run_one` entry and grid loop. ADR 0009's pairing, immutable snapshots,
owned workers, source checks, locks and retry rules remain required. Stochastic
arrival, processing-time and outage ablations preserve other input components.
Transport, storage and quality facilities are factory structure in this model;
their comparisons require explicit factory/scenario cases rather than the former
module switches that could imply free transport or unlimited buffers.

One integer-tick commit resolves Buffer rankings first and then concurrent
Machine, Quality and AGV proposals. AGVs move by cells and explicitly interact
with ports. Capacity is enforced at actual transfer, without invisible destination
reservations or implicit infinite PRE/POST facilities. Failed simultaneous claims
reject all competitors; accepted movement and processing advance in parallel.
Quality inspection locks the station; scrap/output outcomes preserve original
demand identity across replacement attempts. Completion means qualified demand,
with a separate static truncation outcome and a fixed dynamic horizon.

Energy execution, Gantt rendering and video export are outside this slice. Energy
fields stay portable in factory authoring but do not affect execution. They do not
create implicit charging agents. CP-SAT has no grid adapter and must not be
presented as a supported provider for this contract.

## Observation, recording and UI

Render, verbosity and recording are independent controls. Live rendering consumes
committed in-memory records. `run.json` always stores metadata, frozen inputs and
outcome; `trace.jsonl` is optional and stores the trajectory. No second render log
or sidecar timeline file is required. The read-only viewer shares Studio drawing
components in an independent window. Closing it detaches the consumer and resumes
headless work; pausing it pauses simulation advancement at tick boundaries.

Offline playback reconstructs the recorded view without executing the engine.
Execution audit is a different operation: it reexecutes semantic commands and
compares recorded effects. Keeping these operations separate prevents a visually
plausible rerun from silently replacing the original trajectory.

All three learning providers use the same production transition contract. The
adapter's intermediate decisions consume zero physical time, so credit assignment
uses elapsed physical time rather than counting decoder requests. Policies and
critics, optimizer state, samplers and RNG states are checkpointed. Framework
imports remain optional. Old checkpoint action/observation encodings are rejected,
not interpreted as new grid actions.

## Integration gates

Validate hand-calculated transitions, collision/capacity rejection, quality and
replacement ownership, outage boundaries, input round trips, rendering/recording
independence, seek/reverse and live pause/single-step behavior. Train and load real
models; compare visual samples, terminal values and recorded state with physics.
Keep full repository regressions visible while dependent experiment interfaces
are migrated. Do not retire tests merely to suppress failures. A feature-branch
smoke run establishes engineering behavior only, not a research-performance claim.
