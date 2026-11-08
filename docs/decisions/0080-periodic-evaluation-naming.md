# 0080 — Periodic evaluation naming

Date: 2026-11-08
Status: accepted for implementation by user request.

Supersedes the public intermediate-validation naming in ADR0057 and ADR0079.
Use `periodic_evaluation` for the authored schedule, data split, runtime phase,
output directory, and tracking prefix; use `periodic_evaluation_cases` for its
case count. The native runner module is `reservation_periodic_evaluation`.
General configuration/domain validation remains validation. This is one current
contract, without a second V2 tree or parallel legacy recipe set. Historical
source-frozen runs require their archived reader; no in-place migration occurs.

The fixed namespace80000, four conditions, case ordering, RNG isolation, selected
best rule and complete last continuation are unchanged. All45 V2 learner recipes
evaluate every50 episodes, three cases per condition, and retain best/last only.
The separate final80-case test protocol and existing final `evaluation` directory
remain unchanged. Intermediate evaluation still serves model selection, so test
worlds must remain disjoint. Renaming is not permission to tune on final tests.

Progress, resource accounting, W&B metrics and curve step bindings must consume
the same periodic-evaluation phase. Historic decision records retain their terms.
