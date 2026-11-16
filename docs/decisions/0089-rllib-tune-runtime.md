# 0089 — Official RLlib algorithms within the unified Tune runtime

Status: user-authorized migration; implementation and acceptance incomplete.

The migration continues in the original Tune worktree on
`codex/rllib-tune-runtime`. Existing uncommitted Tune work, historical branches,
frozen benchmark sources and running experiments remain preserved. This decision
does not authorize publication, merging or formal experiments.

## Framework boundary

For new runs, official Ray 2.58 PPO/DQN Algorithms, Learners, EnvRunners,
Connectors and replay replace native training. This supersedes the physical-time
discounting and native update/recovery rules in ADR0074, ADR0082 and ADR0083,
and the phase resource changes in `0087-official-tune-execution.md`.
Use the new API stack and official algorithm defaults, recording resolved values
and explicit overrides. Keep the simulator independent of optional frameworks.

Preserve semantic self-organizing decisions, Base and implicit SL attention/concat.
Expose actual decision makers through MultiAgentEnv and use official module mapping
for sharing, without merging agent trajectories. PPO retains public global value
inputs and local/social actor inputs. DQN has no global critic. Model extensions
may implement candidate scoring and action masks, but not replacement algorithm
losses. First-version MARL uses one factory per EnvRunner.

Use FIFO Tune scheduling with resident training and evaluation components;
training and evaluation run sequentially, with evaluation internally parallel.
Remove phase resource callbacks, common bootstrap and stage actor reconstruction.
Profiles declare total resources and driver, Learner and EnvRunner reservations,
including idle components. There is no online resizing or performance calibration.
Default zero retries includes RLlib worker restart settings, not only Tune.

## Scientific and recovery contracts

For complete cases, report J = makespan/5000 + mean_tardiness/300 +
(1000/300)*(1-final_passing_rate). Base and both SL variants accumulate observed
interval reward -(delta_time/5000 + delta_overdue/(300*N) +
1000*delta_failed/(300*N)). Do not sum duplicated agent rewards to obtain factory
cost. Zero-time decisions add no elapsed-time cost. E1a keeps its own reward.

The calibrated drain is a computational truncation boundary, with a legal final
observation for bootstrap. It is not a production deadline and adds no V3
unfinished penalty. Incomplete cases have no complete J. Fixed evaluation panels,
worlds, seeds and action seeds remain explicit. Best minimizes mean J over a
complete fixed argmax periodic panel; an incomplete panel cannot qualify.
Optuna keeps its three unprioritized objectives, separately from J.

Official RLlib/Tune checkpoints own model, optimizer, target, counters and
explicitly enabled DQN replay persistence. Best/last reference those checkpoints;
there is no second model storage implementation. Recovery is at official commit
boundaries, without promises of half-episode simulator or bitwise RNG recovery.
Old framework checkpoints cannot resume under the new framework. Retain the
approved censored-search adapter and prohibition on retrying failed search IDs.

## Source integration and acceptance

The Tune worktree is based on main `cd7074b`. V3 `be0f54a` and concat `1e91c81`
are integration sources, not authorization to restore native schedulers or V3
deadline penalties. Their ADR0084/0085 refer to V3 workload/launch decisions;
the Tune worktree's formerly same-numbered drafts are now ADR0087 (execution)
and ADR0088 (shared calibration). Only draft identifiers and references changed.
The pre-renumbering drafts remain in the migration safety snapshot; V3 ADR0084,
ADR0085 and concat ADR0086 retain their exact source bytes. New runtime decisions
supersede the V3 native training, deadline and checkpoint-selection rules above.

The new performance baseline is direct RLlib+Tune with the same research model,
algorithm configuration, fixed work, evaluation, checkpoints and tracking as the
SmartSOM entry point. Three alternating pairs per representative case must meet
the user-approved 5% steady and end-to-end median limits. Initial startup is
reported separately but remains in end-to-end time. Old Tune acceptance remains
incomplete and is not evidence of RLlib acceptance.

## Known DQN exploration boundary

An isolated Ray 2.58 probe showed that the default exploration implementation
uses nonzero Q values as random-action weights. Artificial all-zero Q values
raise an invalid multinomial distribution error; mixed zero/nonzero values can
exclude a legal zero-Q action. This does not establish occurrence during normal
SmartSOM training.

The user explicitly chose to retain official exploration and normal initialization.
Do not add custom epsilon-greedy, upgrade Ray for this probe, change rewards or
shift Q values to avoid it. Verify real research models through exploration,
greedy, online next-state selection and target calculation. A concrete legal-action
failure requires evidence and discussion of official alternatives before changing
this decision. Forward probes alone do not establish complete training acceptance.

## Open checkpoint integration issue

A separate official MultiAgentCartPole probe on Ray 2.58 used string agent IDs,
shared DQN, the new stack and `store_buffer_in_checkpoints=True`. Saving through
`save_to_path()` and restoring through `Algorithm.from_checkpoint()` preserved
the module weights and allowed another training iteration, but replay contained
256 environment steps before saving and zero immediately after restore. This
does not meet the required replay recovery contract. The general Algorithm new
stack state path omits replay, while the older stack contains its save flag logic.

A public `get_state`/`set_state` extension delegating replay state to its official
methods was authorized after source checks and an isolated Ray 2.60 reproduction
also showed lost replay. Keep Ray 2.58. The extension stores replay in the same
official checkpoint root, using the official writer, reader and buffer state
methods. A complete restore missing replay must fail. Explicit partial state
operations may exclude replay; replay itself is an indivisible component.
Restoring into a populated Algorithm creates a fresh official buffer before
loading state, because the official episode loader populates only empty buffers.
Do not claim complete checkpoint acceptance until replay contents, priorities,
indices, counters and subsequent training have been verified. This decision is
separate from the accepted Q=0 exploration boundary; exploration stays official.

For the frozen module inventory, the extension materializes implicit zero sample
counts in official MARL replay state. Ray 2.58 initializes these counters with
integer defaults but restores missing counters with list defaults; a warmup
checkpoint otherwise fails on its first sample. Explicit zeros preserve the
counter values without changing sampling or replay implementation. Dynamically
adding modules after restore is outside this runtime's frozen configuration.
The user also authorized materializing missing per-module minimum-priority trees
using the official `MinSegmentTree` class in this same state extension. Ray 2.58
restores their missing-key factory as `SumSegmentTree`, breaking sampling after
empty or warmup checkpoint recovery. Only absent trees for the frozen module
inventory are created with official neutral values and the serialized root tree's
capacity; existing priorities and trees remain unchanged. This workaround depends
on Ray 2.58's replay state schema and must be revalidated before any Ray upgrade.
Further recovery testing exposed a separate eviction-path defect: the
official multi-agent prioritized buffer does not restore its reverse
sample-to-tree index map. Continued training can fail when saved episodes are
first evicted. Passing a short continuation before eviction is insufficient;
do not increase capacity or shorten the test to hide this boundary. The user
subsequently authorized a thin official replay subclass overriding only
`set_state`: after the official restore, reconstruct `_sample_idx_to_tree_idx`
as the inverse of the restored `_tree_idx_to_sample_idx`. This is an explicitly
approved dependency on an internal Ray 2.58 attribute, not an additional replay
algorithm. Select the subclass explicitly in the resolved replay configuration;
do not silently replace other replay types. Verification must cover repeated
eviction, priority updates, continued learning and a second checkpoint restore.
The official whole-Algorithm restore is not transactional: a failed restore may
already have loaded other components, so callers must terminate that instance.

### Fixed sampled DQN evaluation

The user confirmed preserving sampled DQN panels with the official
`evaluation_config.epsilon=0.05` override. This is an explicit scientific
configuration choice, not RLlib's default evaluation behavior (which disables
exploration). Argmax cases call official inference without exploration; sample
cases call the official exploration implementation with the fixed evaluation
epsilon. The training epsilon schedule is unchanged. Dynamic or implicitly
inherited evaluation epsilon is rejected for fixed sampled DQN panels, because
the evaluation EnvRunner clock depends on worker assignment and prior sampling.
No custom epsilon-greedy implementation is introduced.
