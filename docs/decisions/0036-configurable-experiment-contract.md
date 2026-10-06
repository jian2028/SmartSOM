# 0036 — Opt-in configurable experiment contracts

Date: 2026-10-06
Status: Development execution contract

This extends ADR 0032–0034 without replacing V4 authoring or the V3 simulator.
Default configurations retain their historical serialization and scientific
identity algorithms. New options select `smartsom.configurable-experiment/v1`;
authors may also select that marker explicitly. Raw author paths, documents and
hashes remain in frozen provenance. The new semantic identity hashes resolved
worlds, role bindings, implementation selectors, weights, optimizer settings,
seeds and evaluation cases. Output/logging, numerical-thread/concurrency ceilings,
retention mode and resolved file locators are execution or provenance details.
The legacy identity and historical continuation contracts are not rewritten.

Algorithm policies may declare a full observation implementation and matching
full candidate-network implementation. These are strict built-in selectors,
versioned and pinned to normalized source hashes. Physical-job inspection choice
is an explicit per-instance boolean. Drivers, native learners, spawned samplers,
model readers and Tune reconstruct the same instance; no launcher-wide install
is required. Eight aggregate fields remain critic-only. Released original routes
are public reference data; latent durations, unreleased orders and terminal
history remain excluded from physical-job features. Existing process-local
installation remains a historical compatibility entry point.

Validation accepts a strictly increasing nonempty `updates` list with
`every_updates: null`, or the existing periodic schedule. Milestones must fit
the training budget. Both use the existing ranking, tie and patience rules.
Opt-in best evaluation can record an authoritative `no_eligible_best` scientific
outcome after a completed training run with validation and `save_best` enabled.
Primary final results and selected checkpoint remain null. An optional last
diagnostic uses the same frozen cases and action mode, has an explicit label and
is excluded from ranking. Missing or corrupt selected artifacts and engineering
failures remain errors. Workflow completion is separate from scientific selection;
the native and parent batch ledgers can complete with this unselected outcome.

`checkpointing.retention.mode: latest_full_and_best` opts into immutable owned
generations. A complete update publishes one authoritative manifest atomically
before changing aliases or collecting older owned generations. Latest FULL
contains optimizer, replay, simulation and RNG continuation; selected best is
inference-only. An explicitly requested initial/control model is an additional
inference dependency. This mode replaces periodic/keep-last history accounting;
default historical trees are never adopted or deleted. Readers and publication
share a process-safe lease. Recovery trusts the manifest rather than stale
aliases. Payload checksums, semantic identity and ownership are verified before
publication and deletion. Windows records its directory-fsync limitation.

Tune retains one latest native adaptive commit with its selected inference
dependencies. The pinned Ray checkpoint manager receives `num_to_keep=1` per
opt-in trial before actor staging; legacy trials in a mixed cohort retain their
previous policy. Native, adaptive and Ray persistence tiers can hold replicas
of the same latest FULL boundary; bounded retention is not a claim of a single
physical copy across all tiers. Recovery preserves their verified dependencies.

Calibration off and fixed scheduling retain ADR 0033–0034 resource admission:
one unmeasured trial first, then observed memory permits refill toward the frozen
global/per-file ceilings. The global ceiling is not a sum across files or groups.
Fixed scheduling prevents allocation resizing; it does not promise dedicated
hardware, immediate full concurrency or bypass resource reserves. Public
`batch-run` still performs engineering smoke and source-freeze checks.

Qualification is engineering evidence only. Formal scientific execution requires
an integrated main commit or explicit tag and a separately admitted frozen plan.
