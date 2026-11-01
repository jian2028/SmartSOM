# 0063 — Explicit action representation and POST pickup credits

Status: implementation in progress; acceptance and hosted CI are separate evidence.

Supersedes the POST supply definition in ADR0058 and the implicit direct action
space in ADR0059. Historical experiments continue with their frozen source. This
change does not add a simulator version selector or a second training driver.

Algorithm components v2 requires `action_representation.default: direct|rules`.
There is no runtime default. Role overrides precede role.stage overrides; entity
specific overrides are unsupported. Centralized assignment accepts only explicit
direct actions. Policy bindings, learner selection, social inputs, network
ownership and traffic remain separate choices.

Rule projection resolves named heuristics to currently legal semantic physical
candidates before observation extensions and learner sampling. Rule lists are
ordered alias priorities; aliases yielding the same physical action have one
representative and no extra probability votes. Builtin IDs retain the cloud/server
catalog's stage-local codes even when selecting subsets. Special STOP and empty
Dispatcher WAIT remain distinct actions under their existing permissions and
physical timing. No Buffer service WAIT is introduced. Unavailable-only subsets
fail explicitly, never silently execute another policy.

Builtin rules read the common current public fields. Custom rule factories declare
`smartsom.rule/v1`, return a callable `(facts, decision, candidates)`, and return a
legal candidate identity or None (unavailable). They receive detached immutable
public facts and candidates; they cannot mutate physics. Rules are deterministic,
stateless functions; stochastic/stateful control belongs in the existing policy
interface. External files/bundles use the existing explicit loader, source freeze
and identity contract. Trusted Python is not a security sandbox.

Projection retains original candidate-linked social inputs by physical semantic
identity and rekeys only the representative. Custom observations run afterwards,
so they see the actual rule candidates. The builtin 48-field encoder uses slots
45:48 for rule codes, preserving the original 1..7 binary encodings; higher custom
ordinals extend the high-order component instead of colliding. Output heads and
PPO/Double DQN update, reward and owner routing remain unchanged. Full alias audit
maps do not enter learner rollout/replay memory.

POST free credit is max(0, R + P - C): R is physical finished POST stock including
loading locks; P is a distinct already-started local machine job, including paused,
blocked and local READY continuation; C is accepted unfulfilled pickups. Unstarted
PRE jobs never create pickup credits. Completion transfers P to R; pickup completion
removes R and C together. Existing failure cancellation remains. Negative balances
emit explicit evidence and fail the reservation invariant. PRE admission and
centralized ready-task allocation retain their distinct semantics.

Continuous empty-unassigned vehicle age, current feasible downstream machines,
POST credit components and current nominal processing work are current public
fields. No future faults, arrivals or hidden quality are read. Historical scores
under the old observation/protocol remain separate scientific conditions.

Replay defaults off separately for train, validation and evaluation. `first` means
first episode per fault condition within a training trial, or first case per
method/fault condition within each trial/checkpoint/decoder evaluation panel.
`all` records each case. Stable episode/case paths support partial-work restart;
receipted evaluations are skipped. Local rule summaries always record counts and
explicit denominators. Optional full mappings/traces stay local; W&B receives
bounded summaries, not every decision or entity snapshot.

Original cloud PPO and server DQN protocols/source indexes are preserved under
`docs/archive/rule-credit-20261101` with full local source archives in ignored
backup. New authoring definitions create new experiments; no old checkpoint is
silently loaded into the new architecture. No formal campaigns are launched here.


Prespecified episode checkpoints can be evaluated on the test panel after the full
training budget, without misusing validation for held-out test results. Checkpoint
nodes are unique, bounded by an episode budget, retained at safe commit boundaries
and evaluated with their own recorded progress. Native train/evaluate may explicitly
set `seconds: null`; optional per-case wall protections remain separately recorded.
Other finite-budget workflows reject null before launch. These paths preserve the
server's no-wall-stop protocol without substituting a very large finite number.
