# 0036 — V3 arrived service and empty reconsideration

Date: 2026-10-06
Status: Superseded by [ADR0037](0037-v3-service-and-learning-liveness.md).

This records the initial implementation at `978fe3a`, including its subsequently
corrected arrived-only behavior and `/2` contract. It is preserved as decision
history, not current authorization or supported semantics. ADR0037 restores the
original enroute opportunity and records the reviewed phase/clock/restore repairs.
The separate PR10 numbering conflict will be resolved in that branch.

Supersedes ADR0035's one-shot empty-episode latch and in-transit empty-trigger
requests, and narrows ADR0021's optional loaded redirects after arrival.
Simulator V3 and semantic action/observation identities remain v3.1. Historical
source and evidence retain their original contracts. Physical metadata records
`nonexclusive-intentions/2`; old models and continuation states require retraining
rather than silent compatibility relabeling.

At a proposal boundary, freeze loaded vehicles eligible for unloading at their
current target: they must have arrived, have no active service, have a legal
quality/operation destination, satisfy port-bound slot capacity and receive
existing queue admission. Shared slot counters and one-service-per-port constraints
bound this set. These vehicles receive no optional Dispatcher request and have
priority in the existing service phase. Other loaded vehicles may redirect.
New zero-time arrivals cannot displace the frozen admitted drop. Capacity is not
reserved in transit; loading/unloading still consumes one physical tick.

An empty vehicle with an existing target receives a Dispatcher opportunity only
when arrived and its owner remains physically empty under ADR0035's full-owner
empty definition. This condition persists across boundaries even after an exact
same-target selection. There is no unconditional timer, forced useful-target
heuristic, new mask, new action or reward change. A target with physical work
retains commitment. Transit vehicles receive no Dispatcher request. Exact same
target preserves progress and queue timestamp; a changed port, including within
the same owner, remains a real reroute with a fresh timestamp. Manual mid-trip
retarget attempts through direct physical APIs still fail explicitly.

Restored opportunities do not guarantee a useful greedy action. Source identities,
nonexclusive intentions, matching, quality, capacity and existing movement physics
otherwise retain their contracts. This decision authorizes no formal experiment,
release, push or merge.

The companion generic DQN cadence repair truncates parallel waves at optimizer
physical-tick boundaries without changing frozen environment ordering or the fixed
weight version within each wave. Serial and parallel trajectories need not match.
Replay sampling remains without replacement, with the existing full-batch gate;
impossible replay capacities below batch size now fail configuration validation.
Details and audit limits: [repair record](../v3-dispatch-learning-liveness.md).
