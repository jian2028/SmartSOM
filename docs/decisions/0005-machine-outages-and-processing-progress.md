# 0005 — Machine outages and processing progress

Date: 2026-09-07

Status: Accepted; implements the Week2 item 7 contract.

## Context and authority

Breakdowns make a task's start-to-completion span different from its processing
requirement. Availability is independent of occupancy: a failed machine can
remain occupied by its paused task. This decision extends ADR 0003's event phases
and clarifies ADR 0002's planned dynamic-input storage. It preserves the existing
arrival and processing input authorities and ADR 0004's independent pairing.

## Decision

`MachineOutagePlan` is an immutable finite union of half-open integer intervals
per semantic machine ID. Overlapping, duplicate and touching intervals merge.
Empty plans are valid. `scenario.machine_events` alone enables fixed input or
`exponential_uptime_v1`; omission/null disables it. Resolution validates and
materializes the plan before simulator construction and run-directory allocation.

Profiles explicitly specify each participating machine's mean uptime and positive
integer uniform repair range, plus an exclusive failure-start cutoff. Unlisted
machines do not fail. Uptime starts at zero, then at the last repair; it includes
idle calendar time. No failures are generated while down. Exponential draws are
rounded upward to at least one tick. A failure before the cutoff retains its full
repair interval; the cutoff does not truncate the simulation. Only the existing
`machine_events` seed is consumed, through versioned per-machine SHA-256 identities
and local RNGs. No simulation-time sampling occurs.

The module owns only immutable event inputs and indexed interval queries. The
engine owns availability, occupants, accumulated processing, active segment
starts, time, event consumption and trace. On breakdown, the task keeps its first
start, machine and mode; the engine pauses it and cancels its active completion.
Repair automatically resumes the remaining work on that machine before any new
decision. Resumption is not another dispatch. Migration, restart, rework and
repair-worker allocation are outside this contract.

Same-tick phases are completion, breakdown, repair (including automatic resume),
reveal, release, then decision. Each phase orders semantic IDs. Tick-zero outages
precede the first decision. Completion at the start of an outage wins and does
not pause. Machine availability during intermediate phases follows consumed
events, rather than anticipating unprocessed events at that tick.

The calendar tracks one active completion per processing operation. Cancelled
heap entries cannot appear in pending events, advance time, trigger a decision,
or complete work. All existing transition checks remain, extended to processing
conservation, paused occupancy and completion lifecycle. Legitimate repair waits
auto-advance; all-jobs-complete termination ignores unrelated future events.

## Observation and replay

`MachineState.availability` adds `up/down`; `OperationStatus` adds `paused`.
Unfinished `actual_processing_ticks` is null. A completed operation exposes its
net executed ticks in subsequent decision snapshots. Future breakdowns, repair
times and true remaining work remain private. The two existing decision triggers
are unchanged; there is no new machine-event notification or completion callback.

The cooperative policy information boundary still receives only `DecisionContext`.
Input plans, traces and run artifacts are privileged evidence, not policy inputs.
SPT uses currently feasible nominal durations and never waits for another mode's
machine to recover. CP/full-static scenarios reject any enabled machine-event
declaration, including an empty plan. Offline fixed-break CP in acceptance is an
independent reference tool, not a dynamic solver provider.

`Simulator`, action replay and schedule replay accept the same `machine_events`
keyword. Every transition uses `step()`. Schedule spans retain first start and
final completion; pause/resume trace reconstructs active segments. Validation
subtracts downtime from spans, checks immediate resumption, and reserves the
machine throughout the span. No arbitrary intermediate idling is supported.

## Evidence and compatibility

`realized_machine_events.json` is independently reusable and includes canonical
content digest plus optional generation provenance. The existing arrival-only
`realized_events.jsonl` remains unchanged. Processing tables remain separate.
This replaces the earlier planned interpretation of one mixed realized-event
file; scientific input components can be imported and ablated independently.
The executed trace remains unified and adds breakdown/repair/pause/resume facts.

Imported content is used directly; changing run seed or provider never resamples
historical provenance. Named-seed values and existing workload/arrival/processing
generators remain unchanged. No-outage schedules, actions and complete traces
are preserved; observation objects gain only the two documented fields.

Failures retain realized inputs, delivered observations and available trace.
Unfinished runs report null makespan. Reference/source snapshots and tests prove
bounded engineering behavior; they do not establish scalability or research
performance. Batch, debug logging, logistics and dynamic CP remain later work.
