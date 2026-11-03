# 0070 — Central rule controller through existing components

Status: implementation; verification and formal experiments remain separate.

Extends ADR0059's custom learner contract and supersedes the role-only reporting
assumption of ADR0061 for explicitly selected custom learners. Built-in shared and
independent Base/SL ownership, ADR0063 rule projection, physical protocol, world
sampling and immutable historical source requirements remain unchanged.

A central policy may use the Base reservation protocol and answer its existing
role/stage requests using one global actor and critic. `control: base` selects the
protocol; production components select decision authority and learning. This does
not enable learning on the separate direct-assignment `control: centralized` path.

A custom learner factory may expose `validate_parameters(**params)` and
`network_ownership(options)`. Compilation validates and reports the custom learner
without coercing its parameters into the native PPO/DQN schema. Built-in reporting
retains its prior defaults. This is a trusted component extension, not a trainer
framework or plugin registry. Custom factories still declare sharing/role-parameter
capabilities and observation/network schemas.

Observation instances may opt into detached static topology via
`needs_topology = True`. Geometry is cached per simulator episode and is read-only;
ordinary Base/SL facts and numeric inputs do not gain topology. No future demands,
fault schedule or realization samples are included. An optional policy
`begin_episode()` resets episode-local memory before any decision, while existing
complete-update checkpoints remain the only durable continuation boundary.
An optional `episode_metrics()` supplies namespaced policy instrumentation.

The controller-comparison package implements C1-inspired fixed-layout central
rule PPO and Justin heuristic adaptation. Central PPO uses native gamma-one
returns, per-decision ratios, role-wise advantage normalization and shared task
reward; it deliberately differs from original C1 reward/action/update semantics.
Network capacity is reported, not matched. Formal performance comparisons and
claims of pure architectural causality are outside this implementation.
