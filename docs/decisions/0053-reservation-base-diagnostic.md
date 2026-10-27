# 0053 — Reservation Base development diagnostic

Date: 2026-10-25
Status: User-authorized implementation; scientific acceptance remains separate.

For the explicitly selected `smartsom.base-reservations/v1` contract only, this
supersedes ADR0037's nonexclusive intentions, ADR0038's blind Output and ADR0039's
fixed-window shipment objective. Historical V2/V3 recipes, checkpoints and
recordings keep their existing behavior. ADR0050's training potentials are not
enabled for this Base. The approved source integration includes performance PR27
at `263e36849fe256fd86dda085974bcc8b6be94a9d`.

Use the existing `ProductionSimulator` kernel with `contract="base"`. A separate
staged protocol yields detached semantic requests; policies return entity-ID
actions and cannot change physics. Machine chooses job×mode. Dispatcher chooses
a source/destination station. Pre/post accept applicants iteratively or STOP;
Input fills available quota without STOP. Rejections prevent repeated requests
within a finite negotiation round. Positive retry intervals and changed local
opportunities wake uncommitted vehicles. Accepted targets persist until service
completion or the specified breakdown cancellation.

Pickup quota counts resident pre+machine+post jobs minus existing pickup
commitments. Delivery quota is station capacity minus resident jobs and delivery
commitments. Neither quota allocates a particular job, dock, waiting cell or pre
slot. On-site job–AGV pairs and unload vehicles are selected iteratively without
STOP. Service locks persist for one physical tick, including inventory and slot
occupancy. A same-timestamp target breakdown cancels an unfinished unload before
transfer, while an operation completing over that preceding interval succeeds.
Adjacent compatible operations remain on their machine, with a new mode choice.

Final Output unload ends every order, reveals its accumulated quality and applies
one failure penalty, without rework. The sole world reward is
`-(delta overdue time + lambda * delta final failures) / N`, with gamma one.
No reward is added for zero-time negotiation or multiplied by decision counts.
Completed total reward equals `-(mean tardiness + lambda * final failure rate)`.
Safety truncation does not produce a complete tardiness or joint-cost result.
PPO uses a critic bootstrap for a safety/budget cut; genuine delivery termination
has zero bootstrap. These are distinct from ADR0039's finite-window terminal.

Five parameter groups use typed mean/max/count set summaries, candidate-specific
features and stage heads. STOP has its own head. Actors obey the approved local
information scopes; the training critic sees only aggregates of released public
state. Hidden fault calendars, future arrivals and undelivered quality are absent.
The first diagnostic uses PPO; this does not select it as the paper's final
algorithm or claim the value-learning route has been compared.

Rules and random use the same semantic candidates and fixed Mover. Random samples
all feasible production choices, including job×mode and legal STOP. The rule is
EDD/standard-mode processing, nearest ready pickup, distance/load delivery and
greedy feasible acceptance. The Mover reserves the next physical interval jointly,
can displace idle vehicles, and checks vertex and edge-swap constraints. It does
not change production targets. Idle event jumps cannot cross service, operation,
fault/repair, arrival, retry or deadline boundaries. Recorded movements are actual
execution, not predictions.

The approved two-scale scene, conditional truncated exponential outages and
finite arrival generator have new explicit four-file schemas/entrypoint. They do
not reinterpret legacy four-file files. The overnight supervisor freezes source,
inputs, calibrated drain limits and matched per-scale physical-tick budgets.
It evaluates fixed final checkpoints on held-out cases, rather than claiming a
validation-selected best checkpoint. Validation/test reports remain separate.

The user explicitly authorized a new branch, push, server deployment and a bounded
24-hour development diagnostic. This is not a formal experiment from integrated
main, a merge authorization, a new social-learning run or a performance claim.

Details and commands: [Base diagnostic](../base-diagnostic.md).
