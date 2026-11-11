# 0086 — Explicit classification of implicit social fusion

Status: user-confirmed implementation scope, 2026-11-11; verification is separate.

Extends ADR0076's candidate-linked social observations with two **implicit**
variants. Supersedes its requirement that every ISL network use attention; the
attention variant retains its architecture, initialization and behavior.
ADR0084/0085's physical, reward, learning, ownership and panel settings are unchanged.

The component parameter `social` declares `kind: none` for Base or
`kind: implicit` with `fusion: attention|concat`. `kind: explicit` is reserved
and rejected as unimplemented. Neither variant changes rewards, legal actions,
or physical constraints. Historical `social_enabled` inputs still map to
Base/implicit attention; combining the historical switch with `social` is an error.

Base stays unchanged, as explicitly confirmed by the user. Concat feeds raw
social scalars and validity bits directly into the existing scoring MLP after
the base features. It has no learned social encoder, attention or relation MLP.
It keeps candidate→job→downstream relations, including each supplied job at
Input/POST acceptance. Flattening jobs and stations into unrelated global sets
would lose this association and is not permitted.

Concat sequences are ordered by semantic entity ID within each relation (job,
station and competing vehicle), padded to explicit bounds, and masked. A slot
is the rank in that ordered relation, not a persistent physical entity ID or
an action index. Invalid payloads are zeroed; masks distinguish absent entries
from real all-zero observations. Overflow fails rather than silently dropping
information. The historical attention ordering is unchanged.

For each map the generated bounds cover the maximum compatible stations for any
operation, the complete AGV fleet, all original orders for Input, and POST capacity
plus one in-machine job. Small uses 2/10/300/5, Medium 5/25/750/5, Large 10/50/1500/5.
The same declaration is required on the paired built-in observation/network.
Changing bounds changes the model and checkpoint identity; it is not a runtime knob.

Raw concatenation does not imply fewer parameters. The Input branch can be much
larger because it retains all job/downstream relations. Parameter counts and
engineering performance must be reported rather than claiming capacity matching
or attention-removal speedups. This change does not establish learning quality.

Existing `isl-{shared,independent}` paths remain the attention experiments, with
fusion explicit in YAML. `isl-concat-{shared,independent}` adds concat experiments
for Small, Medium and Large. No training is launched by this change.

An active experiment must keep its original source and frozen inputs. Its
checkpoint is not resumable under new source merely because Base or attention
math is unchanged. The original V3 source directory remains at its running commit;
development proceeds on the same V3 branch in a different worktree.
