# 0042 — Explicit initialization, random streams and directory selection

Date: 2026-10-12
Status: Authorized local implementation; CI/integration review pending

Extends ADR0025, ADR0033, ADR0036, ADR0039 and ADR0040 without replacing the V3
core, reward equation, checkpoint lifecycle or continuous queue. Default stream
derivation, group selection and serialization retain their old semantics.

Explicit training.groups supports compatible inference models and fresh models
learning together. Shared per-role model sources give equal weights/encoder
state; group policy/action RNG remain independent. Optional world/environment/
policy roots and the policy-based training environment path freeze exact named
derivation. Fixed held-out world/environment roots remain policy-independent.

Initial control is an additional authoritative native stage using the frozen-case
evaluator and initial inference dependency; Tune reuses its existing control
implementation. Experiment task_reward exposes the unchanged ADR0039 objective
to rules and controls. Oracle statistics stay outside policy inputs.

Directory --id is an ordered whitelist; --tag reuses output.tags with AND
semantics. Only selected dependencies are compiled. Resolved IDs/order/stages
and strictly earlier batch.requires enter execution identity, not child science
identity. Resume uses the frozen plan and does not accept a new selection.

Multiple learning stages each reuse the existing Tune/broker queue. Completion
is a strict stage barrier while intra-stage order is admission preference and
can skip resource-blocked heads. No later-group smoke/calibration/training runs
before its barrier. Children retain the whole parent's digest and exact stage
entry identities. Failures require explicit retry; pauses hold the next group.
Rich/monitor preserve all staged children plus the active tuning view. Existing
observed replay/process memory high-water marks and live reserves remain active.

[The interface guide](../experiment-public-interfaces.md) gives exact flags and
schema fragments. No experiment-specific launcher or large data plan is committed.
Publication waits for the main repair and green main, then feature CI and
independent review. Engineering checks do not establish research acceptance.
