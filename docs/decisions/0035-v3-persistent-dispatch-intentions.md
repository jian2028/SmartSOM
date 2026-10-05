# 0035 — V3 persistent Dispatcher intentions

Date: 2026-10-05
Status: Authorized implementation; verification reported separately.

Supersedes ADR0020's exclusive source quantities and optimal pickup matching,
and ADR0021's nonpreemptive automatic geometry trips. This remains V3, with
action/observation identities v3.1. Historical source, evidence and states remain
unchanged and require their original source; incompatible models, continuation
states and semantic replays are rejected.

Empty AGVs select a structurally reachable pickup owner/port, regardless of
inventory or competing intentions. Intentions do not claim quantities or Jobs.
The source-count feature previously called reserved now counts intentions;
it is distinct from physical occupancy and can exceed source capacity. Actual
loading exclusively binds a Job. Buffer selects its ordered legal prefix; source-
local earliest-arrival AGVs receive it in order, with private seeded exact ties.
Historical matching descriptor names are readable aliases for first_arrival,
never selectable old runtime behavior. Port service and capacity locks remain.

There is no Dispatcher NO_REQUEST, KEEP or WAIT. Zero legal targets means no
policy request; physics advances. Exactly one target is a forced non-actor choice.
Choosing the exact same owner and port preserves the trip and queue timestamp.
A changed intention leaves the old queue and rejoins with a fresh timestamp.

An empty AGV outside active loading receives at most one decision per physical
tick when its target becomes physically empty: Machine PRE, POST and internal
Job must all be empty; input and inspection use present physical work only.
Future arrivals and the external FIFO never arm the event. This applies both
in transit and after arrival. Selecting an already-empty new target arms one
decision at the next boundary. Consuming it with the same target does not rearm;
arrival alone does not rearm. Observing physical work resets the episode latch,
so a later transition to empty can trigger again. Changing target creates a new
intention. There is no periodic or congestion trigger. Active services are locked.

Automatic geometry travel retains a deterministic four-neighbor BFS path and
unit-step progress. Each physical tick updates actual coordinates; retargeting
starts there at the next boundary without refunding elapsed ticks. Obstacles
remain solid; AGV roads and port queues remain abstract and can overlap.
Automatic matrix overrides must agree with geometry. Manual matrix values do
not define an intermediate position: mid-trip retargeting fails explicitly
rather than inventing a route or teleporting. Arrived/manual-zero service remains
representable; a broader manual-matrix retarget contract needs a separate choice.

Reward and Social Learning remain unchanged. No further observation features,
formal science launch, publication or model compatibility claims are implied.
