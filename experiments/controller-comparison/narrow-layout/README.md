# Central rule PPO and Justin on narrow layouts

This package adds two current-protocol comparators to the existing
`rule-dqn/narrow-layout` Base/SL study. It does not reproduce the original C1 or
Justin simulator. Baseline source references are main `7ea4243`, Morris
`8eb1ee6` (C1), and Justin `80d40ba` (`all_rules_clearance_v2`). These references
identify design inputs; run manifests record the actual integrated implementation.

## Select a method without changing the entry

An Experiment references one Algorithm. Choose `algorithms/central-rule-ppo.yaml`
or `algorithms/justin-rule.yaml`; keep Scenario and the execution entry unchanged.
Base/SL retain their existing Algorithm files and implementation. `control: base`
means the **reservation protocol**, not distributed neural-network ownership.
The Morris recipe has one central actor/critic, reported by `check --verbose` as
`central=global(1)`. The separate `control: centralized` direct-assignment protocol
is not used or changed.
Tracking identifies the method as `morris_central_ppo` with centralized ownership,
rather than inferring distributed execution from the reservation protocol name.

```sh
uv run smartsom check experiments/controller-comparison/narrow-layout --verbose
uv run smartsom check experiments/controller-comparison/narrow-layout \
  --id medium-central-ppo --select 'train_seeds: [4101]' --id medium-justin
```

The first-wave execution command below is a prepared command, not an experiment
launched by this change. Formal execution requires integrated main or an explicit
source tag, a suitable environment and separately authorized launch:

```sh
uv run smartsom run experiments/controller-comparison/narrow-layout \
  --id medium-central-ppo --select 'train_seeds: [4101]' --id medium-justin \
  --output-root runs/controller-comparison/narrow-layout --wandb offline
uv run smartsom resume RUN_DIRECTORY
```

Switch to the desired tracking mode explicitly. Initial selection excludes small
and seeds 4102/4103; the authored inventory retains them. Directory execution
without selectors requests **all eight tasks**: six training trials and two
rule evaluations. The evaluation task's existing `seed-N` directory is scheduler
bookkeeping, not a Justin training seed; its 40 worlds come from the frozen test
panel. No learner or checkpoint is created for Justin.

After the first wave, `--resume-missing` with the same output root can select
additional authored trials under unchanged frozen source. Do not append these
trials to the old Base/SL batch, change its manifests, or relax source checks.

## Scientific inputs

Medium and small reference their original normal/low/medium/high Scenario files
and the existing `common/*-test.json` panels with unchanged checksums. Orders,
arrival and fault settings, world generation, tick/drain budgets and action
permissions follow the corresponding Base recipe. Every central model trains
across the four conditions in balanced cycles, not as four separate models.
The complete central inventory is seeds 4101/4102/4103, 1000 episodes, checkpoint
every 50 episodes, and the final episode-1000 argmax evaluation on 40 fixed cases.
Justin executes the same cases once per map with deterministic choices.

## Centralized Rule-Selection PPO

The central actor answers each existing decision request in native order. It
uses exactly the Base/SL ordered rule subsets, processing-mode expansion,
alias merging, legal candidate projection and STOP/WAIT permissions. No old C1
supply/commit/exclusivity heuristic prunes the candidate pool. SharedMover still
owns road movement and the current protocol owns admission and service.

Public fixed-layout input comprises:

- time, completed count and released-job aggregates: count, minimum/mean slack,
  overdue/rush counts, accumulated steps, remaining operation count and public risk;
- each machine's down/processing/blocked/occupied state, elapsed/nominal work,
  mode and capability indicators, and its job aggregate;
- each AGV's position, loaded/service state, arrival age, target-owner indicators
  and cargo aggregate;
- each buffer's capacity/unlimited indicator, quota, accepted commitments, ready
  count and inventory aggregate;
- static blocked-cell grid and owner port count/mean location, current native
  local context, decision-stage and owner indicators.

Entities are ordered by stable IDs. All five native role groups share one actor;
this is not five identical networks. Medium and small have separate learned
models and fixed input dimensions. The layout fingerprint checks topology,
entity identities and machine capabilities; equal vector lengths do not establish
checkpoint compatibility. No future release/fault or unrealized quality samples
enter the actor. The optional topology interface is evaluated only for components
that request it; Base/SL keep their existing observation path.

A same-tick prefix records `(role, entity, stage, rule identity, physical action)`.
The network consumes the corresponding stage/owner indicators and selected
candidate features (including the stable rule code), not temporary candidate
indices. The prefix is frozen with each decision, cleared on time advance and
explicitly reset for every episode, including repeated episodes starting at the
same tick. A 32-unit GRU encodes it. One candidate-scoring MLP and one independent
48-field native-critic MLP each use two 64-unit tanh layers. Inference never reads
the critic channel. There is no cross-episode recurrent memory.

PPO uses full-episode gamma-one return-to-go, native truncation bootstrap,
per-decision clipped likelihood ratios and role-wise advantage normalization,
then ordinary shuffled minibatches from all roles. There is no role-balanced
resampling, GAE, joint-action likelihood ratio or 256-step C1 update. Forced
single-candidate decisions retain the native critic/normalization treatment but
have zero actor contribution. Loss is actor + SmoothL1 critic - entropy bonus;
gradient norm is capped at 1. Adam learning rate is 0.0003, epochs 3, batch size
256, clip 0.2 and entropy coefficient 0.01. Reward is the same current task reward
as Base/SL, scaled by 0.001 for learning. Old C1 shipment/progress shaping is absent.

The first public observation determines network dimensions. Empty initial
checkpoints preserve the initialization seed; populated checkpoints include model,
Adam state, update count, shuffle RNG and sampling generator. Existing episode
transactions and source freeze provide continuation. `policy_metrics` records
parameter count, decision count and accumulated inference seconds (packing,
forward and selection, excluding first model construction and observation
projection). These timings are instrumentation, not a hardware benchmark.

Do not interpret current PPO-versus-DDQN results as isolating control architecture.
Even after matching the learner, information access and network capacity remain
explicit differences. Parameter counts are reported rather than forcibly matched.
No superiority or convergence claim follows from engineering smoke tests.

## Justin adaptation

Justin uses direct legal candidates with deterministic ranking, not a learned
rule-selection policy. Adaptations from the original implementation are explicit:

| Stage | Current behavior |
| --- | --- |
| Machine | Normal mode first, then shortest nominal processing time, stable identity tie break |
| Empty AGV pickup | Ready/potential-supply tier, excess inbound vehicles, queue pressure, greatest completed job step, distance and route continuity |
| Loaded AGV delivery | Destination fullness, route continuity, incoming vehicles, occupied owner ports, distance |
| Acceptance | Current BaseRule default acceptance order; existing protocol admission remains authoritative |
| Buffer service | Current BaseRule EDD order |

Current requests select an owner, not an old `(owner, port)` intention. Consequently
queue/occupancy preferences are owner-level. Accepted current commitments replace
old intention-only admission, and pickup quota supplies the potential-supply cue.
There is no extra fleet allocation/masking pass and no new WAIT. Old inspection
preferences are omitted because current physics uses final Output inspection.
The old clearance mover is replaced by the common SharedMover. These are protocol
adaptations, not evidence of equivalence to Justin's original branch.

## Verification boundary

`tests/integration/test_controller_comparison.py` covers actual decisions/updates
on both maps, returns/clipping, prefixes, initial and trained checkpoint recovery,
layout rejection, opt-in public information, Justin priorities, configuration
alignment, relocated frozen bundles and first-wave selection.
The suite also exercises public CLI resume from an incomplete committed run and
multi-stage 250-tick rollouts with exact subsequent-update restoration.
Existing component,
rule-action and native learning suites remain regression gates. Local engineering
results and review notes belong under ignored `artifacts/`, not formal experiment results.
