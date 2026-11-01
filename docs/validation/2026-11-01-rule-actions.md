# Rule-action integration: engineering evidence

Scope: follow-up to integrated PR31/main `1e66a85`; branch
`codex/rule-actions-protocol`. This is an implementation/acceptance record, not a
research result or authorization to run the migrated campaigns.

## Local evidence

- Fixed-source local unit + related native suite: **1,589 passed, 4 platform skips,
  1 warning**, 197.73 seconds, macOS CPU. After the custom-rule ordinal stability
  correction, its affected rule/native subset passed **31 tests** in 16.53 seconds.
- Earlier specific coverage includes all four Base/ALL × PPO/Double DQN combinations
  with rule projections, mixed direct/rule stages, independent Machine ownership,
  160-tick engineering rollouts, updates, evaluation and full checkpoint restore.
  Continued weights, random action trajectories, target networks and replay sizes
  are checked; these are not convergence or algorithm-quality measurements.
- Protocol tests ported from the frozen server/cloud handoff cover PRE exclusion,
  R/P/C atomic transfer, blocked/paused/local READY credit, loading locks, deficit
  invariants, unassigned-idle clocks, downstream feasibility and public-information
  boundaries. New tests cover explicit mode selection, overrides, subsets, special
  actions, alias priority, custom rules, frozen multi-file reload and invalid output.
- Replay tests cover off/first/all, completed-case resume, interruption followed by
  a distinct preserved attempt, catalog navigation and per-condition training first
  recording. Prespecified test checkpoint evaluation is distinct from validation;
  explicit no-wall-stop training survives complete-update continuation.
- Ruff lint/format, whitespace and full CI ownership inventory pass locally.
  Hosted CI and final independent review remain pending.

## Frozen-source physical comparison

The independent reference implementation is the server source at `b269f55`, whose
shared protocol/rule files match the cloud handoff. The original historical Small
map fixture, eight orders, seed123, deterministic fixed Rule, default STOP and new
credit/public-field options produce the following complete normalized trace hashes:

| Condition | Physical ticks | Delivered | Trace SHA-256 |
| --- | ---: | ---: | --- |
| normal | 142 | 8 | `3890f35fc7ff0f25b3f18bfb34415965bced5bea0f7fb4a6b21632feb1911e0f` |
| high | 167 | 8 | `b93f6534d12e7dfc15b25996955efff4e582f94d60168637e1bd72bfd500162a` |

Current code matches these full decision/observation/action/event/movement/time
traces exactly after removing additive serialization fields `social`, `inputs`
and `rule_audit`. This is a physical and observation comparison under fixed rules,
not proof of identical historical neural-network training. Prior-main traces
intentionally differ after the approved POST-credit change and remain historical.

The original 20-machine factory and workload canonical hashes match the current
`original-20-machine` scenario. All four imported rule-credit evaluation worlds
regenerate with the original full scenario digests and exact action seeds.

## Archive and boundaries

See [source index](../archive/rule-credit-20261101/source-index.json). Full source
archives are ignored local backup files. Cloud source reconstruction matches its
entire recorded Git tree. Old runs/artifacts stay in their original locations.
Historical checkpoint evaluation/continuation has not been exercised here.

Current native model factories omit unused historical heads. Fresh initialization
therefore need not reproduce the old model's RNG draw order, even with the same
seed. No automatic old-model import or claim of bitwise training reproduction is
made. New experiments retain scientific world/rule definitions while using the
current learner lifecycle and record their new source identity. Original PPO/DQN
protocols have different evaluation arrangements and are not a pure algorithm
comparison. Full-capacity long-run memory, cross-host/CUDA continuation and physical
phone rendering are not established by these short macOS checks.

No formal experiment has been launched. Temporary scripts, traces, checkpoints
and test output stay outside the PR.
