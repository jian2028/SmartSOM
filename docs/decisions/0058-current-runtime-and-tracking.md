# 0058 — Current runtime, explicit research modules and tracking

Status: implemented; engineering acceptance and PR CI are recorded separately.

Supersedes the active legacy-runtime selection in ADR0017, ADR0020, ADR0025,
ADR0031 and ADR0053–0057. Historical records and frozen experiments retain their
source identities and are executed with archived source, never reinterpreted.

The current physical world is shared by Base negotiation and centralized direct
task assignment. Base negotiation, final inspection, reward and Mover semantics
remain those of ADR0053. Native-v2 is the sole built-in Base observation layout.
PPO and Double DQN support the same Base/social input arms. Control, observation
and network modules are explicit experiment inputs with frozen source identities.

Centralized task assignment reserves job, vehicle and destination capacity before
travel. Only currently ready jobs are assignable. Each zero-time decision refreshes
legal candidates; advancing time runs the shared physical transition. Invalidated
destinations release reservations; loaded jobs remain on their actual vehicles.

Current public facts are detached and read-only. Base keeps its existing local
fields, SL explicitly adds peer context, and custom observations may select global
current public fields. Future arrivals, hidden quality and future faults are not
observation inputs. Optional libraries remain outside the simulation import path.

Training/evaluation default to W&B online, with local records authoritative.
Startup failure requires explicit offline selection; noninteractive startup fails
closed. Mid-run tracking failure does not cancel training or its durable commits.
Offline continuation produces separately identified upload segments. Source,
optimizer and replay restoration remain independent of cloud tracking identity.

Source snapshots and lockfiles preceding this change live in the local backup
archive. Existing experiment directories are not moved or removed. Old checkpoints
are not automatically imported into new source. No scientific performance claim
or formal experiment is authorized by engineering tests of this change.

## Launch, identity and monitoring refinement

One compiler serves file, directory and short-name check/run. Scientific matrices
expand before the shared admission queue. Foreground is the default even in the
server profile; a selected profile may only set operation fields. Precedence is
program defaults, Experiment, explicit profile, explicit CLI. Science overrides
are advanced, visible and frozen. No implicit host/profile discovery is used.

Execution sidecars retain operation history separately from science and full
frozen-file checksums. Resume validates current source and complete continuation;
historical controls are rejected. Manifests define catalog facts and unique IDs;
indexes are reconstructible. Same-name/prefix ambiguity is an error.

Local train/validation/evaluation events feed W&B and optional TensorBoard.
Method and selection distinguish evaluation series; axes are physical ticks,
training episode or case index. The terminal shares random-state isolation with
tracking. Bash/zsh completion, config diff and result navigation reuse public
resolution and never create experiments while inspecting them.
