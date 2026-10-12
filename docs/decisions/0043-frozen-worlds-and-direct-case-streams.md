# 0043 — Frozen worlds and direct case streams

Date: 2026-10-12
Status: Development execution contract

This extends ADR0036 and ADR0042. Historical V3/V4 recipes and v1 seed streams
retain their derivation and serialization. An explicit `legacy_policy` selection
uses seed-streams/v2 and `named_seed(policy_seed, 'training:{index}:{episode}')`.
The existing `environment` and `policy` selections keep their environment domain.

V4 may instead select `smartsom.frozen-experiment-inputs/v1`: one factory,
one optional training world, and ordered validation/evaluation cases. Each world
is a strict `smartsom.frozen-production-world/v1` document containing a complete
native ProductionScenario and data provenance. Case IDs are unique per phase.
Each input declares its dataset seed and actual environment seed; each evaluation
case additionally declares an actual policy root. The scenario must match that
environment seed and the selected factory. Native physical validation applies.
Frozen inputs cannot mix with workload/matrix generation, derived world or
environment roots, authored environment physics or phase replication overrides.

Training clones the complete frozen scenario, substitutes the selected episode
seed and applies the existing maximum-tick bound. It does not reconstruct a
partial world or resample declared outages, durations, quality or travel tables.
Evaluation constructs native inference policies with the existing weight identity,
then reseeds each rule or model action stream using the case root and the named
`policy:<group>` domain. Case roots do not reinitialize weights, select another
optimizer, or mutate live training RNG.

Source documents are read and hashed from the same byte snapshot once per plan.
Physical worlds, file hashes, provenance, stable case IDs, actual case roots and
episode domains enter the opt-in semantic identity. Detached recipes, run records
and model packages retain their input/seed contracts. Full continuation compares
that identity before loading learner, simulator, collector or RNG state. Inference
weight initialization remains separate from continuation; compatible same-role
weights can initialize distinct sharing groups and held-out evaluation worlds.

This adds no experiment-specific generator, callback, optimization schedule or
second runner. Qualification is engineering evidence only; formal experiments
still require integrated main or an explicit tag and their own frozen protocol.
