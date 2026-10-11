# 0037 — V3 service and learning liveness

Date: 2026-10-06
Status: User-authorized repair; independent review corrections pending re-review.

Supersedes ADR0035 only for persistent reconsideration after arrival, and narrows
ADR0021's optional loaded redirects when unloading is feasible. The original
enroute empty-episode opportunity remains: once per empty episode while travelling,
with work resetting its latch. Automatic rerouting starts from actual coordinates;
manual mid-trip target changes still reject and roll back. After arrival, an empty
owner permits reconsideration each boundary, regardless of the consumed latch.
Exact same target preserves progress and arrival timestamp. Different ports of
the same owner remain real reroutes. There is no timer, new mask/action, reward
change or guaranteed useful greedy choice.

Simulator V3 and action/observation v3.1 identities remain. Physical dispatch
metadata is `nonexclusive-intentions/3`, rejecting both published `/1` and the
unpublished initial repair `/2`. Model-loading entries reject old metadata;
continuations now save the full physical contract and direct restore checks it
before any state mutation. Missing or differing contracts reject. Public resume's
existing implementation-identity guard remains. Old states/weights are not relabeled.

Machine decisions retain the frozen source-supply boundary. The common coordinator
resolves actual Machine START proposals before generating final Dispatcher requests.
This supersedes ADR0020's shared pre-START Machine/Dispatcher observation boundary.
Dispatcher observations reflect actual START and capacity at that same tick;
new processing supply still counts only at the next boundary. No clock advances
between these phases. Semantic replay verifies the same phase order.

Feasible arrived drops after START are protected from optional redirects, including
slots freed by the chosen job in a multi-slot PRE. Actual bound-slot capacity is
checked again at service. The combined proposal API validates legal proposals,
resolves START first, then removes redirects for newly feasible drops from the
accepted command; it does not apply those obsolete redirects. Advanced callers
can explicitly use `resolve_machines()` before selecting Dispatcher targets.

Existing port admission retains arrival order and private seeded ties. Across
ports sharing slot capacity, earlier arrival wins, with independent identity-seeded
ties grouped by owner and arrival. The same ordering is used in actual drops.
Slot counters and one-service-per-port constrain both predictions and service.
New zero-time arrivals cannot replace a protected drop. Blocked and incompatible
vehicles retain legal redirect opportunities. Travel creates no slot reservation.

Parallel DQN waves stop at the next optimizer tick and the earliest per-group
scheduled target-copy tick. Target clocks, cursor and ticks remain saved/restored.
Warmup, replay full-batch gating and without-replacement sampling remain unchanged.
Invalid replay capacities below batch size now reject. Stable environment order
and fixed weights within a wave remain; serial and parallel learning trajectories
need not match. Bootstrap and truncation censoring are unchanged.

This decision authorizes no formal experiment, push or merge. Details and
verification: [repair record](../v3-dispatch-learning-liveness.md).
