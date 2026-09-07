# ADR 0007 — Finite buffers, blocking and exact execution replay

Status: Accepted for Week2 item 9.

This extends ADR 0006. Its unconditional immediate delivery and machine release
apply when buffers are disabled or all capacities are infinite. The rules below
supersede those assumptions only for enabled capacity limits; ADR 0005's actual
processing lifecycle, event ordering and information boundary remain intact.

## Configuration and physical ownership

`FactorySpec.buffers` contains immutable `MachineBuffers(machine_id,
pre_capacity, post_capacity)` values. Zero means no waiting position; a positive
strict integer limits job count; missing/null means infinite. Unspecified machines
are infinite, and explicitly all-infinite entries normalize away. Unknown machines,
duplicate entries, unknown configuration fields, negative/boolean/float capacities
fail during input validation. Input/output have infinite space. Processing positions
are separate from pre/post capacity, and pre/post cannot borrow from each other.

`scenario.buffers: {kind: limited}` alone enables capacity rules. It is independent
of `scenario.transport`. Python entry points `Simulator`, `replay` and
`replay_schedule` accept `buffers_enabled=False`. Fixed capacities use no RNG and
preserve the six existing named seed domains. CP rejects explicit buffer enablement,
even if every capacity is infinite. No finite-buffer or transport solver is added.

The immutable `BufferModule` provides capacity and occupancy/reservation queries.
The engine's existing `TransportExecution` now serves both real movement modes:
AGV and instantaneous transfer. It exclusively manages positions, bookings,
reservations and handoffs; `Simulator` still owns the clock, calendar and sole
step loop. There is no additional state machine or generic hook framework.

## Admission, reservation and unloading

AGV `Transport` may book an eligible destination even when it is full. If a
positive prebuffer has an unreserved slot at booking, the trip immediately owns
that reservation until unloading consumes it. Otherwise there is no reservation.
Reservation holders never lend their slots. Enroute trips without reservations
do not acquire one when space opens.

Arrival and unloading are distinct. A vehicle that cannot unload holds its job
and stays occupied at the destination. Arrived unreserved vehicles use new free
capacity before any subsequent policy action, ordered by actual arrival tick,
then `(agv_id, job_id, transport_sequence)`. Destinations are processed in stable
machine-ID order. Reservation owners retain exclusive access even when arriving
later. Positive-prebuffer delivery always enters the prebuffer; a free processing
position does not bypass a full queue. Processing selection remains non-FIFO.

Zero prebuffer does not reserve the machine position. A job can travel toward a
busy/down machine but unloads only when that position is free and the machine is
up. It then occupies the machine awaiting explicit `Dispatch`; unloading never
selects a processing mode. No repair/unloading estimate is exposed to policies.

Without AGV, enabled buffers introduce `Transfer(job_id, destination)`. The job
must be ready and the destination must accept it now; validation and movement are
atomic. No waiting request, binding, reservation or future transfer is created.
This mode does not require a matrix. It still has explicit input, prebuffer,
machine, postbuffer and output positions, including movement for a same-machine
successor and the final exit. Both movement actions preserve semantic destinations.

## Completion and occupancy

Dispatch does not reserve postbuffer capacity or require free post space.
Completion ends actual processing immediately. It publishes actual net work and
leaves `OperationStatus.COMPLETED`, even when the job cannot leave the machine.
If postbuffer has space, the job moves there automatically and frees the machine;
otherwise it holds the processing position in the `blocked` machine-holding phase.
This interval is never added to processing time.

When post space opens, an unbound completed holder automatically enters it,
regardless of machine up/down. A vehicle may take a completed holder directly
from its machine with either zero or full positive postbuffer. A booked holder
stays at its original pickup location until actual pickup, even if post space
opens meanwhile. Actual pickup releases the machine. Direct Transfer uses the
same source release rule instantaneously.

A zero-prebuffer job awaiting dispatch may be rerouted to another eligible
machine. Pending self-delivery is invalid. A completed job whose next operation
uses that same machine must still be moved back once; direct zero-to-zero
transfer may atomically release and reload that position. Pending or completed
jobs can be taken from down machines; processing and paused jobs cannot.
Faults pause only processing work. Availability remains independent of holding
phase: empty, awaiting dispatch, processing, paused, or completed-blocked.

The calendar retains completion → breakdown → repair → pickup → arrival/delivery
→ reveal → release. After current-tick calendar processing and capacity-releasing
actions, automatic postbuffer handoffs and arrived-vehicle unloading run to a
stable state before a decision. Every handoff is invariant-checked. They neither
advance time nor dispatch processing. Zero-time physical transfers add no tick.
The two decision triggers and both wait actions retain their existing contracts.
All jobs must reach output before termination; makespan uses final actual exit.

## Baselines, observations and failure

SPT/first-feasible prioritize processing as before. Their AGV policy records
`buffer_admission_rule: immediate_capacity`: positive prebuffer must offer a
reservation now; zero prebuffer must be up, free and have no other inbound trip.
Taking the destination's own completed job out and returning it for the successor
counts as vacating that position. The existing idle-destination reroute filter
continues. Direct transfers use stable job/destination ordering. These are policy
filters: scripted actions retain broader legal early delivery and loaded waiting.
No safe policy move uses `WaitNextEvent`; with no future event this explicitly
fails. It does not assert that the engine lacks other legal actions.

A real deadlock has no legal physical action and no future event. Its error
includes job positions/holders, waiting vehicles, capacities and reservations.
No transport cancellation, swapping, spill space or automatic recovery is provided.
Neither conservative filtering nor finite buffers guarantee deadlock-free routing.

Decision snapshots add capacities, pre/post occupants, reservation owners, machine
holding phases and waiting vehicles. Active-trip data exposes arrival calculated
from the fixed matrix, not actual future unloading. Future arrival/reveal events,
repair times and unfinished actual work remain hidden. Snapshot fields are frozen
values and tuple collections. Trace records actual reservations, block/unblock,
arrival/wait/unload and direct transfers alongside the existing processing events.

## Versioned execution evidence and replay

Finite-capacity AGV and all direct logistics use `smartsom.execution-schedule/v2`.
`ExecutionSchedule` contains processing, full trips, separate `TransportArrival`
records, `ScheduledTransfer` records and `TimedAction` entries. Trip/transfer
sequences are contiguous from 1 in booking/transfer order. Action order is a
separate contiguous sequence for **all non-wait actions**, with their submission
ticks. This preserves same-tick dispatch/booking order and reservation ownership.
Input array order is irrelevant. A completed trip's `delivery_time` is actual
unload, and vehicle occupancy lasts until then. No future delivery is fabricated.

Prevalidation checks structure, coverage, references, strict times, processing
intervals, vehicle continuity, matrix travel and action associations. One replay
policy follows explicit action order, using waits only to reach specified ticks;
every physical transition passes through `step()`. Capacity and position legality
are checked there. Final verification compares the complete schedule and output
makespan. Missing moves, extra unloading delay and wrong same-tick actions fail;
replay never repairs a timetable or compresses idle time. The runner uses the
same conversion/verification as the independent replay API.

Unlimited AGV retains v1 processing/trip schedules and original complete trace,
including when buffers are explicitly enabled with all-infinite capacities.
Disabling buffers retains legacy behavior and input/seed goldens. Direct logistics
adds real Transfer actions; no trace-equivalence claim to direct processing is made.

`RunEvidence` saves effective capacity configuration, rule identity, actual views,
incremental trace and successful complete schedule. Progress separately counts
processing completions and output jobs. Deadlock/provider/replay/writer failures
retain available partial evidence and do not report a successful makespan.
See [item 9 acceptance](../validation/buffers.md) for independent hand calculations,
matched external evidence and its limits. Shared pools, swaps, spill buffers,
automatic recovery, batch, debug logging and learning are outside this change.
