# 0003 — Independent arrival timing and policy visibility

Date: 2026-09-07

Status: Accepted; implements the Week2 item 5 contract.

## Context and authority

ADR 0002 separates workload structure from materialized dynamic events and
distinguishes physical release from information reveal. This decision fixes the
first concrete event module. It supersedes the earlier planning suggestion that
release timing should live inside `WorkloadInstance`; it does not rewrite ADR
0002 or alter existing workload generators and content digests.

## Decision

`ArrivalPlan` is an immutable table covering every job exactly once. Each entry
has strict integer `0 <= reveal_at <= release_at`. Reveal exposes the whole job,
including its serial chain, modes, nominal durations and exact release time.
Before reveal, neither that job nor its operations, candidates, counts or next
event time appear in `DecisionContext`. Resources are known factory inputs.
Unknown and unrevealed operation IDs produce the same dispatch rejection reason.

`scenario.arrivals` is the sole enabling declaration. Omission/null means off;
there is no second enable flag. It selects either a fixed JSONL table or the
`uniform_release_v1` profile. Resolution materializes a complete plan before
creating a simulator or run directory. Its digest is independent of the workload
digest. Root seed authority stays in the run; only the derived `demand` seed is
consumed by the arrival generator. Imported seed metadata is historical provenance.

The concrete `ArrivalModule` owns timing projection and immutable event inputs.
The engine owns runtime state, event consumption, time and trace. This is
composition of actual behavior, without a module registry, generic hooks or a
second simulation loop. No runtime RNG or configuration parsing enters the core.

The scenario selects a decision trigger:

- `dispatch_available` (default) returns decisions only with legal dispatches.
- `arrival_event` also returns one notification after same-tick reveal/release
  events, possibly with no candidates. Completion-only ticks without candidates
  auto-advance. Ordinary same-tick dispatch decisions still occur as needed.

All completions precede all reveals, then all releases; each phase sorts semantic
IDs. Policies see only the settled tick. Zero-time facts initialize state without
extra arrival trace, preserving the complete old trace for all-zero plans.
Revealed but unreleased initial jobs permit an empty tick-zero notification.
When none are revealed, initialization advances to the first relevant event.

`WaitNextEvent()` explicitly waits without revealing a hidden event timestamp.
It fails atomically without a future event. `WaitUntil` retains its one-step
semantics: an earlier event can end that wait, and no hidden waiting commitment
survives a returned decision. Empty candidate sets are handled explicitly by
SPT/first-feasible; scripted policies must supply the wait themselves.

Only online providers with `decision_context` visibility can use arrivals.
The existing offline CP provider is rejected even for an enabled all-zero plan.
Full schedule replay remains a privileged validation input, using the same
engine with release checks; it is not evidence of an online policy's knowledge.

## Consequences and limits

`DecisionContext.jobs` exposes immutable domain job descriptions after reveal;
its operation-state collection is filtered to those same jobs. Candidate masks
also enforce release. This is an API information contract for cooperative
policies, not a sandbox for hostile Python introspection or scripts that already
encode future knowledge.

Runs with arrivals additionally retain reusable timing rows and the exact
observations delivered to policy selection. Semantic trace contains positive-time
arrival facts and actual waits. Input files and manifests are privileged evidence,
never policy observations. Failed attempts preserve these and partial records.

Machine events, late reveal, dynamic CP, learning, batch and other dynamic
modules remain outside this decision. No research performance claim follows
from deterministic replay or the small engineering acceptance cases.
