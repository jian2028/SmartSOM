# 0025 — Four-file authoring and explicit detached execution

Date: 2026-09-27
Status: User-authorized implementation; verification is recorded separately.

Supersedes **the daily author-file ownership portion of ADR 0020** for new v4
inputs, and extends **the foreground-only launch boundary of ADR 0024** with
explicit POSIX background execution. ADR 0020's simulator/model ownership and
ADR 0024's default foreground behavior, verified process ownership and
cooperative stop safety remain in force. Existing v2/v3 inputs and saved runs
retain their scientific identities and strict continuation limits.

## Author ownership and detached compilation

Factory, Workload, Algorithm and Experiment are the four daily author files.
Factory retains the v2 schema with additive optional reliability rules. Workload
v3 owns external data, arrivals, Job classes/allowances, independent initial
history and measured content V. Algorithm v2 owns rules/models, stable role and
entity group bindings, observation/network/reward extensions and one explicit
learner/backend. Experiment v4 owns an explicit task, the three references,
policy and independent data seeds, budgets/cases, runtime, orchestration,
logging and output.

Scenario, Composition and Policy remain typed internal objects. A compiler reads
real references once, applies strictly validated selectors/field overrides,
prepares these objects in memory and calls the existing execution boundary. It
must not write temporary author files or invent paths to satisfy loaders. Shared
author files are not edited by CLI overrides. Real source paths, source hashes,
effective documents, overrides, data instances, resolved method identities and
code identity are frozen with a new run.

New single runs are one-entry plans; Factory × Workload matrices use the same
entry driver and task dispatch. They do not inherit the historical Small Study's
fixed H/V/algorithm matrix or report assumptions. A training-seed axis is allowed
only for training tasks; rule evaluation uses actual held-out data replications.
Check compiles the same plan without creating run output or allocating workers,
learners, Ray or performance calibration. Data generation/scoring can still have
substantial computation cost.

## External data and physical boundaries

Faults are an independent Factory extension, not a component of H. Fault streams
use stable machine identities and supplied data domains; available uptime includes
idle time, repair excludes overlapping failures, and processing resumes retained
work on the same machine. Generator rules stay in Factory and realized timelines
stay in frozen run snapshots. Studio must preserve optional reliability fields.

Workload v3 separates content, rush, history and ordering randomness from policy
and environment randomness. One shared segment pool, external slots and per-window
rush plan are reused across V and environment/method conditions. The approved
model's full edit-distance/minimum-coupling V compares before history update.
The new v3 construction uses a reference Mid permutation and a declared finite
budget of legal same-segment, same-class, cross-window exchanges for Low/High,
accepting only strict full-sequence improvements. This replaces ADR 0021's older
minimum/middle/maximum candidate construction **only for Workload v3**; the old
content-recipe generator and evidence are unchanged. Failure to achieve ordered,
separated levels is an explicit data-construction failure, not a reason to redraw
or relabel the pool.

Fractional reference work is preserved before complete physical scaling and
rounding. Normal allowances use declared reference work, operation count and
base terms; rush applies a shorter positive ratio before one physical-grid
rounding. Public rush does not force environment insertion or add an implicit
reward transform. Offline V, statistical initial history, future Jobs and
unrealized events are not Agent observations.

Exact generation is distinct from learner performance calibration. An in-process
cache and saved frozen data may avoid repeated generation; neither establishes
large-pool runtime qualification. An optimization must preserve the full-series
score, frozen pool and acceptance criteria rather than silently approximate them.

## Explicit rules and execution capabilities

Custom rule YAML chooses a registered name/version and parameters. Only explicit
`--extension-module` authorizes Python module loading. Rules receive detached
public semantic choices with named features, return a current legal semantic
action and do not own time or physics. Stateful rules declare reset and finite
JSON state serialization/restoration. Module, rule and declared helper-code
identities are frozen and propagated to owned workers. Missing or changed code
refuses execution/recovery; YAML does not authorize a new import capability.

Observation, reward and encoder registries remain general versioned interfaces.
No special Social file, implicit learner selection or concrete Social Learning
method is introduced. Optional learning/Tune dependencies remain outside the
simulator and framework-free rule boundary.

## Background, control and stage recovery

Background is explicit at launch and supported by the local POSIX implementation
on macOS/Linux; Windows reports unsupported. It starts a distinct process session,
closes terminal input, persists stdout/stderr and waits for the real driver's
registered process/run identity before reporting launch success. The detached
driver persists preparation/calibration/execution failures. A launch handshake is
not experiment completion. Foreground remains the default; monitor remains a
read-only projection of saved progress, not a keep-alive process.

A v4 directory freezes its `smartsom.author-plan/v1` in `plan.json`, detached
per-entry preparations under `inputs/`, status in `batch.json`/`run.json` and
native training/evaluation stage ledgers under `entries/`. Delegated Tune plans
use their existing verified adaptive commits under `performance/`, including the
complete training/final-evaluation phase marker. `control/` lifecycle sidecars
and `logs/` display/detached-output files are independent of scientific inputs.
Execution/display choices and source path spelling are not scientific identity;
data, physical/method contracts and required implementation identities remain
frozen and verified.

One ownership/control protocol covers new single, native batch and Tune entry
paths. Ordinary stop halts new admission, saves at supported safe boundaries and
releases owned execution resources. Timeout does not authorize automatic killing;
explicit force uses verified PID creation identities and ownership and never
closes unrelated applications or shared Ray runtimes. Only verified process exit
permits reporting stop completion. Old active unregistered runs are not
retrofitted.

V4 plans save training/evaluation stage completion. Recovery skips completed
stages, resumes training from a complete native update when its existing recovery
contract permits, and reruns an interrupted evaluation stage. No promise is made
of within-case evaluation continuation or saving an incomplete forced update.
Supported execution/display options may change during recovery; scientific
input, method/code drift or changed data requires a new run. Older v2/v3 saved
runs keep their older recovery restrictions and are not silently migrated.

Performance recommend/auto routes supported learning train-evaluate plans through
the existing calibration/Tune boundary. Recommend stops after producing its
recommendation; auto executes with the measured choice. Rule evaluation uses
native concurrency and rejects this automatic training calibration. Scientific
hyperparameter search, remote resource discovery, shared-cluster management and
GPU/H20/CARC qualification are outside this decision.

A joined probe deadline is recorded separately from a worker or cleanup failure.
At the active measurement deadline, a prior complete, currently feasible baseline
may be recommended with an explicit lack-of-convergence explanation. An incomplete
baseline, ordinary failed candidate or unverified cleanup cannot enable automatic
execution. Performance off disables calibration; explicit Tune execution requires
recommend/auto rather than silently ignoring off.

Local implementation, engineering tests and scientific research outcomes remain
distinct evidence. This decision itself does not assert platform acceptance,
large-pool generation performance, trained-policy quality or formal experiment
completion.
