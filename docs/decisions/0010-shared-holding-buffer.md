# 0010 — Shared holding buffer for AGV congestion fallback

Status: Accepted and implemented for the IDETC integration slice.

This extends ADR 0006/0007 with one global holding resource, using their existing
transport ownership, reservations, loaded waits and same-tick closure. The feature
is present in the frozen IDETC paper-stage source at commit
`729bc692c85781be275f42144ce582449b875d98`; its source digest and limitations are
recorded in `data/reference/holding/source.json`. The old freeze records dirty
state, so this establishes matched file/feature provenance, not byte-exact
identity of a complete historical execution environment.

`FactorySpec.holding_buffer` contains `HoldingBuffer(buffer_id, node_id, capacity)`.
The node must exist in the factory transport matrix. Capacity is null/infinite,
zero/no storage, or a positive strict integer; there is no processing-position
bypass at zero. `scenario.holding_buffer: {kind: shared}` enables it only with
AGV transport. Machine buffers remain an independent switch. There is no new seed.

Python Simulator/action replay/schedule replay take `holding_buffer_enabled=False`.
A holding destination is `TransportDestination("holding", buffer_id="b-hold")`;
job location has kind `holding`. Only ready unfinished work from postbuffer or a
completed-blocked machine may be parked. Input, prebuffer, processing and paused
jobs cannot be parked. A holding job goes to an eligible next machine, not back to
holding. Fully processed jobs go directly to output. No direct Transfer is added.

The same engine logistics component owns positions, vehicles and reservations.
At booking a finite free slot is exclusively reserved. An unreserved trip may
arrive loaded and wait; it cannot borrow reserved space. Free space first serves
arrived waiting vehicles under the existing arrival/AGV/job/sequence ordering.
A bound holding job keeps its slot until actual pickup. No extra clock or dispatch
is introduced by unloading. Reserving/consuming holding capacity has explicit
holding trace records; actual transport events record pickup/arrival/delivery.

SPT/first-feasible still dispatch processing first. Holding is a transport fallback
only when that job has no safe eligible machine destination, and holding can admit
it immediately. Existing shortest-trip and stable ID tie-breaking then apply.
Kernel/scripted actions can park proactively and book a full destination. General
deadlock remains possible and is reported, not repaired by cancellation or swaps.

Current holding capacity/occupants/reservation owners are immutable observations.
Future arrivals/repairs and unfinished processing/quality truth remain hidden.
Final quality inspection still happens only at output. Holding-enabled execution
always uses v2 ExecutionSchedule and the existing step-based replay converter;
actual unload, pickup, complete movements and action sequence must match exactly.

Disabled holding preserves earlier complete action/trace/observation encoding.
Old destinations omit the new optional buffer ID. Enabled holding rejects CP;
this does not add a transport solver or change existing static CP acceptance.
