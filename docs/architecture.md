# Current architecture

[ADR 0058](decisions/0058-current-runtime-and-tracking.md) replaces runtime version
selection with one current physical core. Earlier ADRs and validation records are
historical evidence, not alternative current launch contracts.

## Physics and decisions

`ProductionSimulator` and `PhysicalServices` own released workpieces, machines,
AGVs, capacities, service, movement, faults and physical time. `BaseProtocol`
implements application/acceptance/service semantics. `CentralControl` exposes
machine (job, mode) and transport (job, AGV, destination) candidates. Assignments
immediately reserve resources and destination capacity; same-time commits have
zero physical duration. Advancing time executes the shared physical transition.
Loaded cancellation retains real cargo and vehicle ownership for reassignment.
The optional Gymnasium adapter reports semantic candidates and `physical_dt`.

Public facts are detached, read-only current state. Future releases/faults and
unrevealed quality are excluded. Built-in Base observation uses the corrected
native-v2 layout. SL adds explicit peer context; turning it off retains Base
inputs. Explicit Python observation/network/policy factories are checked and
frozen with source hashes. Different actor observation shapes are allowed; the
built-in value channel has its own defined interface.

## Learning and evidence

Native PPO and masked Double DQN share applicable observation encoding while
keeping policy/value and Q heads separate. Continuation commits complete updates
atomically, including optimizer, randomness, counters and DQN replay/target state.
An interrupted incomplete rollout replays from the last committed boundary.
A weights-only file does not establish full continuation.

The simulator imports no algorithm framework or tracking SDK. Optional Torch,
Gymnasium, Ray, Qt, TensorBoard and W&B dependencies live at their adapters.

## Configuration and orchestration

Factory, Workload, Algorithm and Experiment are scientific inputs. An explicit
optional profile supplies only launch, concurrency, resource, output and monitoring
settings. Precedence is defaults, Experiment, selected profile, explicit CLI.
Science, observation and extension identities remain separate from operational
history; full frozen-file checksums are retained. Resume reads frozen inputs.

`check` and `run` share compilation and short-name resolution. Directory execution
uses the shared native scheduler and global admission ceiling; Ray Tune uses the
same native session. The lifecycle covers calibration, bounded smoke, training,
validation and final evaluation. Explicit stage constraints remain meaningful.
Run manifests are authoritative; catalog discovery is reconstructible and old
results are read-only. Ambiguous names and ID prefixes fail before control actions.

## Monitoring and replay

A local metric event journal feeds optional W&B and TensorBoard sinks. Phase and
physical-step/episode/case axes are explicit. Workers do not initialize duplicate
tracking runs. W&B initialization is isolated from training RNG; startup failure
requires explicit offline choice, and later synchronization failure does not
invalidate local checkpoints. Online resume retains cloud identity; offline
resume creates separately uploadable segments. Default uploads exclude source
repositories and large checkpoints.

Current snapshots feed the existing Studio replay reader. Historical recordings
remain readable; executing or resuming old experiments requires their exact
archived code, extensions and environment. Archive availability and resumability
are separate evidence states.
