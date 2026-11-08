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

Native PPO and masked Double DQN retain separate heads/losses and applicable
shared encoders. Parameter ownership can be shared or independent by role.
Only native initialization is supported. Complete checkpoints include optimizer,
randomness, counters and DQN replay/targets; incomplete rollouts restart at a
committed update boundary. The v4 checkpoint contract rejects historical layouts.

## Configuration and orchestration

Scenario, Algorithm and Experiment are the authored scientific inputs. Scenario
expands into internal geometry/workload models once. Current resolved snapshots
retain the authored Scenario plus frozen runtime data and component sources.
Resume validates these identities. Recompiling a current snapshot uses that
preserved Scenario, not a historical standalone workload authoring branch.

Profiles change operational settings only. Public check/run, foreground/background
execution, batches, Ray Tune, monitoring and current replay use this same runtime.
The current research family is Small/Medium × Base/ISL/E1a-current PPO/Rule.
Algorithm-neutral machine-breakdown Scenarios define all four fault conditions.
Each Experiment embeds its fixed test seeds and physical-world hashes; compilation
freezes generated inventories as runtime artifacts. See ADR0075.

Optional Torch, Gymnasium, Ray, Qt, W&B and TensorBoard imports stay at their
adapters. Importing the simulator requires none of these frameworks.
Historical source/readers and experiment definitions live outside the tracked
tree; existing runs and checkpoints remain untouched. See the archive index for
source recovery. No new formal run or research superiority is established here.

Native batch evaluation uses the phase lease contract in
[ADR0077](decisions/0077-native-phase-resource-control.md): training keeps a baseline
CPU, evaluation atomically grows its existing task slot, and in-flight cases drain
before a reduced CPU lease is released. Requested evaluation concurrency is distinct
from actual admitted workers and from the aggregate queue budget.

[ADR0078](decisions/0078-unified-operational-cli.md) consolidates the operational CLI.
A configure request validates and applies all resource fields in one journal
transaction. Task IDs, scientific task types, aggregate budgets and per-panel
CPU caps have distinct parameters. CLI consolidation does not migrate running
processes or change frozen scientific inputs.
