# ADR 0006 — Fixed-matrix AGV transport and complete execution replay

Status: Accepted for Week2 item 8.

This extends ADR 0002's resource/configuration boundary and ADR 0005's event
order. Original-machine pause/resume and information hiding remain unchanged.
It replaces the earlier roadmap's buffer-first order: fixed-matrix AGV with
unlimited waiting areas is item 8; finite buffers/reservation/blocking is item 9.

## Resource and configuration authority

`FactorySpec.transport` optionally owns `TransportSpec`: semantic node IDs,
machine-to-node mappings, global input/output, capacity-one AGVs and their initial
nodes, and one complete directed travel matrix. Every machine has exactly one
mapping. Multiple logical locations may share a physical node. The matrix uses
strict nonnegative integer ticks, with zero diagonal; asymmetry and zero off-diagonal
entries are supported. Both empty and loaded travel use that matrix.

`scenario.transport: {kind: fixed_matrix}` alone enables execution; absent/null
disables it. An enabled scenario needs validated resources before simulator/run
directory creation. Fixed logistics never samples randomness or adds a seed domain.
Automatic algorithm parameters declare the currently supported
`transport_rule: shortest_trip` and `rerouting_rule: idle_destination`, including
their effective defaults. No separate case/adapter file or plugin registry exists.

## Runtime and decision contract

`Transport(agv_id, job_id, destination)` is separate from processing `Dispatch`.
The destination is `{kind: machine, machine_id: ID}` or `{kind: output}`.
Booking binds one idle AGV and one released, ready job immediately. The job stays
at its source until pickup but cannot be dispatched or claimed again. The AGV
travels empty, picks up instantaneously, travels loaded, delivers instantaneously,
and remains at the delivery node. Booked trips cannot be cancelled or changed.

Locations distinguish unreleased, global input, machine prebuffer, processing
position, postbuffer, AGV and global output. At release a job enters input; reveal
alone does not make it transportable. Machine/mode selection happens at actual
processing dispatch, from an unbound job in that machine's prebuffer. Completion
immediately frees the machine and moves the job to its unlimited postbuffer;
paused operations retain their processing position and cannot be transported.

Pending prebuffer jobs may be moved to another machine eligible for the next
operation. Prebuffer self-delivery is invalid. Consecutive operations on the same
machine still require postbuffer-to-prebuffer AGV service, including empty travel.
Busy/down machines can receive jobs. Only fully processed jobs may go to output.

The engine owns mutable positions, bindings and vehicle state. `TransportModule`
owns immutable resource/route indexes and computes legality. `TransportExecution`
is an engine component called by the existing simulator; it has no clock, policy
loop or independent event calendar. This separates real logistics responsibilities
without creating generic hooks or a second simulator.

Same-tick event phases are completion, breakdown, repair/automatic resume, pickup,
delivery, reveal, release, then decision. Each phase sorts semantic IDs. Immediate
pickup/delivery closure at the current tick precedes the next decision; zero travel
does not introduce artificial ticks. Existing `dispatch_available` now means any
legal processing or transport candidate; `arrival_event` retains its arrival
notifications. Both wait actions retain their one-call semantics.

`DecisionContext` adds immutable AGV states, visible job positions and transport
candidates with empty/loaded durations. A destination machine appears once per
job/vehicle even when several modes use it. Current trip timestamps are calculable
from the fixed matrix and public; future arrivals/outages/repair times and
unfinished actual processing work remain hidden.

SPT and first-feasible prioritize processing. Their transport baseline minimizes
`(empty_ticks + loaded_ticks, job_id, agv_id, destination.kind, machine_id)`.
It considers prebuffer rerouting only from a busy/down source to an idle/up target;
the core permits other legal reroutes for scripts or future policies. Same-tick
sequential decisions can start physically parallel resources. The baseline does
not loop on zero-time rerouting when processing is available.

## Replay and evidence

`Simulator`, `replay`, and `replay_schedule` accept `transport_enabled=False`.
Action replay preserves actual transports and uses `step()`. `ExecutionSchedule`
contains processing intervals and every `ScheduledTransport`, whose explicit
`transport_sequence` identifies an occurrence even for repeated same-tick trips.
The sequence is contiguous from 1 in booking order, independent of array ordering.
Each record includes job/AGV/destination/source, physical nodes, and booking,
pickup and delivery ticks. Occupancy spans booking through delivery.

Before execution, interval checks validate processing coverage, matrix durations,
vehicle continuity, release/processing prerequisites and complete job location
chains through final output. Replay dispatches only after the recorded inbound
transport occurrences: visiting a machine earlier in a same-tick reroute sequence
must not start processing prematurely. Every change still goes through `step()`.
Final verification compares all processing and transportation records and makespan.
No idle time is compressed, no missing trips are supplied, and no modes are changed.

Termination requires every job at output; makespan is the last actual output
delivery. Processing completion is reported separately in the successful summary.
Trace adds empty-start, pickup, loaded-start and delivery records linked to each
trip. Runs save `execution_schedule.json`, actual observations, transport identity
and incremental trace. Progress includes completed operations and delivered jobs.
Execution/replay/writer failures retain available evidence and no success makespan.

Transport off preserves prior processing schedules, actions, full trace and seeds.
Transport on with all-zero travel still has binding, routing and transport records;
only matched dispatches without intentional waits must preserve processing schedule
and makespan. Full-trace equivalence to transport-off would be incorrect.

## Deferred behavior

Finite capacity, destination reservations and blocking belong to item 9. No
advance pickup, empty reposition action, loading delay, collision, path planning,
charging, vehicle outages or transport CP is included. The static CP provider
rejects enabled transport, even a zero matrix. No daily dependency is added.

See [item 8 acceptance](../validation/transport.md) for hand calculations, fixed
source evidence, compatibility goldens and executable checks.
