# 0083 — Explicit DQN target clocks and measured replay intensity

Status: implementation under review; not an experimental result.

Supersedes the implicit mixed target synchronization behavior retained during
ADR0082. The user requested one current implementation and explicit settings,
without a legacy mode or historical checkpoint migration.

`target_update_clock` selects `network_gradient_steps` (default) or
`global_gradient_steps`. The former uses each network's optimizer counter;
the latter uses all networks' optimizer steps. Existing role parameters can
select either, with the same meaning for shared and independent ownership.
Maintained YAML selects `network_gradient_steps` explicitly. Other sampling,
update, reward and physical simulation settings are unchanged.

Cumulative actual sample rows and target copies join per-network insertion and
gradient counters in committed checkpoints and loss telemetry. Initialization
is not a target update. Replay intensity is actual sampled rows divided by
lifetime inserted transitions, including warmup, and zero before insertion.

Historical frozen runs stay bound to their source. This changes the scientific
configuration of new runs; it does not resume an old run under new semantics.
No alternative public entry point or duplicate experiment family is added.
