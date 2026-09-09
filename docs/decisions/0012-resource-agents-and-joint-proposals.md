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

The RLlib provider uses two independent masked PPO modules, the existing
episode materialization, train/run/study entrypoints and optional dependencies.
Checkpoints bind both roles, projection/coordination versions and base structure.
No old IDETC/centralized checkpoint compatibility or cross-structure generalization
is implied. Full item 13 requires actual 4096-round PPO updates/save/load and all
10 paired MARL/SPT evaluations. Random policies can diagnose interfaces but cannot
replace the required learner. Physics or inputs must not change to obtain a pass.

## Framework and checkpoint protocol

`rllib.resource_ppo` keeps one local CPU environment, numerical thread count one,
and the existing gamma/PPO defaults. RLlib counts joint environment steps; each
resource's contribution is also recorded as an agent step, including NOOP. Actor
parameters in both role modules must change and round-trip through independent
module exports. The new resource-checkpoint/v1 schema records each role's initial
and final hashes, the role map, joint/agent/physical counts and coordination version.
The old centralized checkpoint schema and seed recipes remain unchanged.

Ray's protocol adapter does not call reset during construction. Its isolated API
test uses a legal NOOP sampler because Ray's generic checker ignores masks. Ray
requires resource IDs to remain named while checking final rewards/observations;
the adapter returns __all__ termination/truncation and retains those final IDs,
while the underlying ParallelEnv removes live agents. No further actions are
requested. This is a protocol conversion, not a different physical lifecycle.

ResourceCheckpointPolicy has no Simulator reference. The runner acknowledges each
actual step, closes a joint round when needed and streams its ledger. Evaluation
limits/rewards use the same pure accounting helper as ParallelEnv. Joint ledger
write failure leaves already-written physical evidence and the original cause;
it cannot convert the run to success. Full observations are optional, but candidate
mappings and observation/mask hashes are always retained for verification.

Ray 2.58's default episode iterator yields sets of agent IDs. Before its
AgentToModuleMapping, SemanticBatchOrder orders **every** sampling and learner
column by supplied episode order and then semantic agent ID; it does not order
episodes by generated UUID, change values/distributions, or change core arbitration.
This prevents a Python hash seed from assigning the same RNG draws to different
resource rows. The observed failure and the hash-seed control are recorded in the
acceptance document. At that stage deterministic evaluation still failed after
this correction; that original training checkpoint remains failed. The authorized
numerical-unit adjustment below is a separately retained attempt.

## Learner numerical units (2026-09-08 follow-up)

The user authorized analysis, adjustment and another training attempt without a
commit. `ResourcePPOParameters.learner_reward_scale` is an optimization-only positive
finite constant, defaulting to 1 and omitted from legacy serialization at that
value. The revised micro preset uses 0.0001. Scale a fresh learner reward tensor
before GAE; never mutate sampled episodes, raw team rewards, evidence or evaluation.
Critic predictions and bootstrap targets use these same optimization units. The
scale is preserved in algorithm/resolved/checkpoint identity. No observation,
semantic action, physical objective or failure-cost contract above changes.

This addresses verified clipping of squared value errors at Ray's default 10:
raw targets in hundreds or thousands gave zero gradient beyond the clamp. A
positive constant preserves return ordering, but does change PPO's numerical
optimization. The new fixed-seed, fixed-budget development result passes 10/10
paired runs and all replays; it is not a generalization/performance claim or a
formal main-commit acceptance. Original failed evidence remains retained.

## Formal acceptance evidence (2026-09-09)

The item-13 command distinguishes explicit development checks from formal runs.
Formal acceptance requires a clean commit integrated into local main, including
a detached worktree at that commit. Training, every evaluation and the auditor
must use that same source; source eligibility is rechecked before success.
Frozen identities cover the training recipe and complete paired evaluation
inputs, including event plans, visibility and module switches. Each replication
must contain exactly one MARL and one SPT run, with all physical/observation
audits and the five joint replays passing. Neither batch completion alone nor
an old dirty checkpoint constitutes formal acceptance. The actual platform is
recorded; Linux/h20 validation is deferred independently of macOS completion.
