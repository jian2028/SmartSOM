# Current architecture

[ADR0073](decisions/0073-unified-observation-components.md) defines the unified
observation/component boundary and retirement of historical interfaces. Engineering
verification, hosted CI and formal experiment results are distinct evidence.

## Physics and decision mechanisms

ProductionSimulator and PhysicalServices own workpieces, machines, AGVs, buffers,
faults and physical time. BaseProtocol implements application/acceptance/service;
CentralControl provides direct processing/transport assignments and ADVANCE.
Both use the same physical core and traffic mover. Semantic entity/action IDs,
capacity reservations and legal-candidate checks remain authoritative.

## Public information and components

The engine emits semantic decisions without numerical model features. Explicit
observation factories receive a request-scoped public query provider. Local
queries return detached immutable rows; full facility snapshots and static
topology are available when requested. Dynamic data are fresh for each decision,
including same-tick changes. A provider cannot be used to query later state.

V2 service rules reuse the same job-to-vehicle pairing within one request's
derived-facts scope. Input/post acceptance candidates share their owner's frozen
social supply within that observation callback. Neither reuse spans a decision
or a physical advance; candidate order, tie-breaking and numerical fields remain
unchanged.

V2 numeric observations and Torch callers share one packing layout with their
respective array allocation operations. The RLlib environment builds ordinary
numeric fields directly in NumPy, avoiding intermediate Torch allocation and
conversion. Unusual custom values retain the original Torch conversion rules. Shapes,
float32 observations, masks, ordering and fixed-capacity padding are unchanged;
the official connector still owns conversion to model inputs.

Implicit attention embeds input/post acceptance supply once per owner in the
current batch, then shares those embeddings across candidate queries. Concat
retains its ordered candidate layout. This reuse does not span forward calls or
cache tensors across updates; observations, parameters and checkpoint keys remain
unchanged. Floating-point gradient accumulation can differ by roundoff.

Observation components construct statistics and fixed input data; network
components construct learned encodings and masked action scores. BaseInputs holds
Base context, four sets and candidate vectors; implicit SL appends peer relations
without changing Base fields. These numerical layouts belong to the observation,
not the simulator. Custom named inputs are deeply frozen before use.

Named rules resolve to legal semantic actions before observation construction.
Rule, Random, Justin, Base/SL learners and Morris use the same component loader and
source identities. Morris has one global network over Base reservation decisions;
it is separate from centralized direct assignment. Physical legality stays in the
engine regardless of the selected observation or controller.

## Learning, ownership and recovery

Official RLlib PPO/DQN Algorithms own sampling, losses, updates and optimizer
state. Base/SL research RLModules own observations, candidate masks and social
representations. Parameter sharing uses official agent-to-module mappings.
Official checkpoints retain model/optimizer/counters and the approved DQN replay
extension; best/last are references, not additional model files. Resume occurs at
official iteration boundaries, not native exact waves or arbitrary simulator state.
Old native checkpoints cannot continue under RLlib.

## Configuration and orchestration

Scenario, Algorithm and Experiment are the authored scientific inputs. Scenario
expands into internal geometry/workload models once. Current resolved snapshots
retain the authored Scenario plus frozen runtime data and component sources.
Resume validates these identities. Recompiling a current snapshot uses that
preserved Scenario, not a historical standalone workload authoring branch.

Profiles change operational settings only. Public check/run, foreground/background
execution, batches, Ray Tune and monitoring use this runtime.
The maintained recipes cover Small/Medium Base/SL PPO/DQN.
Algorithm-neutral machine-breakdown Scenarios define all four fault conditions.
Each Experiment embeds its fixed test seeds and physical-world hashes; compilation
freezes generated inventories as runtime artifacts. See ADR0075.

Optional Torch, Gymnasium, Ray, Qt, W&B and TensorBoard imports stay at their
adapters. Importing the simulator requires none of these frameworks.
Historical source/readers and experiment definitions are retained under
`archive/`; existing runs and checkpoints remain untouched. See the archive index for
source recovery. No new formal run or research superiority is established here.

[ADR0089](decisions/0089-rllib-tune-runtime.md) replaces the native execution
and [ADR0087](decisions/0087-official-tune-execution.md) stage-resource design.
Ordinary experiments and searches share one Tune Trainable around official RLlib
Algorithms. FIFO admits the resident driver, Learner and training/evaluation
EnvRunners together. Evaluation runs after training with internal parallelism;
idle components still reserve their declared resources. There is no stage actor
rebuild, bootstrap, online resizing or custom performance calibration.

Whole-run pause commits an official boundary before Tune shutdown; restore reuses
the same Tuner and search state. Declarations are logical admission requests, not
OS process limits. [ADR0098](decisions/0098-rllib-core-closeout.md) limits current
acceptance to the core training/control/search loop. ADR0100 adds named Rule/Random
fixed panels as Tune scientific tasks with reserved Ray case workers and the existing
atomic receipts. Learning standalone evaluation,
window/recovery analysis and new recordings are
explicitly deferred; rule smoke, scientific calibration and historical reads stay.

For calibrated searches, [ADR0088](decisions/0088-shared-search-calibration.md)
fixes every candidate to the validated template's drain. Each trial retains the
complete original report and the original calibration identity content, separately
from its changed algorithm. Candidate inputs and provenance are published together
before scientific execution. Ordinary fixed-run calibration matching is unchanged.

Official sampling and update configuration is described in
[training controls](training-sampling.md); native ADR0082 controls are retired.

PR42 Morris/E1a-current now uses a single-agent RLlib PPO adapter (ADR0101).
Its semantic centralized tasks, CONWIP, public global features, flat candidate
scorer and scaled default reward remain. Compact Gym Sequence rows are batched
through public vector/ConnectorV2 interfaces; the official learner and common Tune
checkpoint/panel runtime own training and recovery. It is distinct from the
historical Morris network over Base role decisions.
