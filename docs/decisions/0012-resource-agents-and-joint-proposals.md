# 0012 — Resource agents, public observations and joint proposals

Date: 2026-09-08

Status: Accepted. Projection/Parallel API and actual RLlib training are separate
implementation checkpoints. This extends ADR 0011 without changing its centralized
projection, the simulator, scientific seeds or physical replay contract.

## Ownership and roles

ResourceProjection accepts only known factory resources and public DecisionContext.
Machines are `machine:<id>` agents sharing machine_policy. Enabled AGVs are
`agv:<id>` agents sharing a separate agv_policy. Agents stay present until global
termination, even when busy or down. Without AGV, a Transfer belongs to its
destination machine, or its source machine for final output. No dispatcher is added.

Algorithm configuration owns fixed projection capacities and normalization. Jobs
bind only after reveal in `(reveal_at, job_id)` order; operation chains use explicit
predecessors, modes sort semantic IDs. Episode bindings neither move nor recycle.
Privileged preflight validates hidden input size but never supplies it to a model.

## Resource projection v1

Three groups retain the frozen IDETC observation organization: global public job
and machine summaries, self-resource information, and current candidate features.
The frozen inner wrapper and package selector hashes are recorded in
`data/reference/idetc/resource_observation_source.json`.

Global fields include released/running/ready jobs, completed-operation counts,
binary machine-visit history, job locations/binding and published inspections;
machine occupancy, down state and distinct-job completion counts; input/output
counts, starving-machine fraction and shared holding occupancy/reservations.
Ready means released, not processing/paused and with unfinished work; it is not
an alternative feasibility mask. Starving means up, unoccupied, no held job and
empty prebuffer. Only public engine candidates authorize actions.

Machine-local features include identity, up/down, holding phase, job, chosen-mode
nominal duration, elapsed time since first start and pre/post occupancy/reservations.
AGV-local features include identity, phase, node, bound job/source/destination and
computable pickup/arrival remaining times. Waiting does not expose unload time.
Internal completion events, hidden actual work, repairs and quality draws are
never queried. Hidden probabilities have an explicit missing marker.

Candidate rows encode action type, revealed job/operation/full mode, destination,
remaining operation count, selected nominal duration, quality/base-mode mapping,
empty/loaded travel times, source and public destination-machine state. Routing
rows expose min/max nominal durations on the destination, without choosing a mode.
Output/holding durations are absent. Missing quantities have presence flags.
Times divide by configured time_scale, counts by count_scale; finite capacity uses
capacity/(count_scale+capacity) and a separate infinity flag. No fitted scaling,
clipping or hidden-instance-dependent dimensions. Job/resource references are
one-hot; candidate padding is entirely zero. Float32 conversion must remain finite.

NOOP is index zero. Other indices refer to the current sorted candidate table;
the mapping is recorded every round. For configured J jobs and K execution modes
per operation, each machine has J*(K+1) candidate rows. Each AGV has J*D rows for
the known D destinations. These conservative bounds do not inspect hidden counts.
Overflow fails rather than truncating. Real candidates exactly cover the engine
view, including aggressive transport choices; no SPT safety filtering is applied.

## Joint round and failures

All proposals are decoded against one frozen snapshot before any physical write.
Processing precedes movement; ties use resource ID and semantic action key. Each
job can receive at most one accepted action per round, including zero-time trips.
Conflicts are rejected explicitly, never resampled. The coordinator reads fresh
public candidates before every accepted action, and calls Simulator.step only.
Newly enabled choices require a new round. A clock change/termination expires
remaining old proposals; they cannot execute using a later snapshot.

NOOP is available to active and inactive agents. If all choose it, one
WaitNextEvent is allowed only with ADR 0011's public witness. Otherwise the episode
ends as policy_stalled. Engine deadlock stays distinct. No hidden events, synthetic
ticks or automatic rerouting rescue a policy.

Every resource receives the same negative elapsed-time reward, including initial
auto-advance. Each successful undiscounted return is -makespan; team return is not
the sum over agents. Limits default to 1024 joint rounds and 10000 ticks, after
atomic physical transitions. Failures cost max(tick_limit+1, actual_tick); budget
exhaustion truncates, legitimate stalled/deadlock episodes terminate. Invalid
actions and implementation/nonfinite errors raise; they are not learning samples.

Parallel terminal masks are NOOP-only placeholders and agents becomes empty;
they do not request further actions. Empty postterminal step returns empty dicts;
nonempty postterminal actions fail. Framework reset options are accepted but have
no scientific effect. Construction/space discovery does not reset an episode.

## Evidence and compatibility

Joint ledgers record snapshot/mask hashes, candidate bindings, indices, proposals,
acceptance/rejection, physical actions, trace ranges, reward and outcome. They are
separate from unchanged core trace. Joint replay audits coordination; action replay
audits the entire physical trace; schedule replay audits intervals, transfers,
quality and makespan. A schedule cannot reconstruct rejected proposals or NOOP.

The follow-on RLlib provider uses two independent masked PPO modules, the existing
episode materialization, train/run/study entrypoints and optional dependencies.
Checkpoints bind both roles, projection/coordination versions and base structure.
No old IDETC/centralized checkpoint compatibility or cross-structure generalization
is implied. Full item 13 requires actual 4096-round PPO updates/save/load and all
10 paired MARL/SPT evaluations. Random policies can diagnose interfaces but cannot
replace the required learner. Physics or inputs must not change to obtain a pass.
