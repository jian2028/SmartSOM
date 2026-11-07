# 0073 — One observation boundary and current experiment family

Status: implemented; final review and verification are recorded separately.

Supersedes the observation/compatibility portions of ADR0059, ADR0063 and the
archived experimental ADR0072, and the historical experiment-authoring paths in
ADR0061. Physical reservation, action identity, reward, ownership and complete
update transaction semantics remain unchanged. No formal experiment is launched.

The physical core emits semantic candidates only. Each policy explicitly selects
an observation callable over current public facts and a semantic decision. There
is no native/semantic switch, prepared-field fallback or optional None projection.
Base/SL numerical vectors are constructed by observation components and stored in
immutable BaseInputs; custom components return deeply frozen named inputs.
Trainable encoding remains in the network. Custom PPO packing does not invoke a
Base-specific tensor cache.

Public facts support synchronous request-scoped queries. Each queried row is an
immutable detached value; querying an unmaterialized live view after callback
completion fails. Local observers do not copy facility-wide state. A new provider
is created after each semantic mutation, including multiple decisions at one tick.
Full public snapshots and static topology remain available to global observers.
No future arrivals/outages, sampled durations or private quality draws are exposed.

Rules use named measures and legal candidates, independently of neural vector
positions. Projection preserves aliases, tie-breaking and special actions. SL
constructs relationships from full physical candidates before matching retained
rule representatives. This preserves PRE STOP's complete applicant context.

Morris remains a single global PPO controller over reservation decision points;
Justin is a reusable Rule policy component. The distinct centralized direct-task
mechanism retains Rule/Random/custom policies, without new native DRL support.
Native PPO/DDQN and shared/independent role ownership remain selectable.

Native initialization is the only initialization. Retiring the historical dormant
heads and reward-division compatibility is an explicit changed experimental
condition. The v3 component checkpoint contract rejects previous layouts instead
of silently attempting continuation.

Scenario is the sole authored scene format. Compilation freezes its source and a
current resolved snapshot; internal factory/workload models remain necessary for
execution. Historical authoring formats, migration commands and playback readers
are retired. Old runs remain untouched and use their archived source/environment.

Only the Small/Medium × Base/SL research entrypoints and fixed case panels remain.
Morris/Justin/Rule/Random are algorithm choices, not duplicated historical study
folders. Small current CLI/extension examples are engineering recipes. Old source,
configs and evidence are archived outside the tracked tree, with checksums and a
verified source restoration. Git history is not rewritten.

Remaining-work caching now keys the job step, location and processing progress,
not tick alone. This fixes stale observations when heterogeneous machine speeds
or elapsed work change within one tick. Such cases can differ numerically from
archived source; this correction is separate from native initialization. Matched
current fixed-map traces are checked independently for numerical equivalence.

Current recovery configuration serialization records explicit default thresholds
and windows. Historical default-omission serializers are removed, so scientific
identity is computed from the new complete representation. Recovery arithmetic
and the default parameter values are unchanged.
