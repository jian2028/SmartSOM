# Two-sided AGV controller — local engineering checkpoint, 2026-10-07

## Scope and source

The 24 upstream commits through `bf7bf20` were integrated at merge commit
`3df9f47bb9846cadc793c9e37a6376555569ad9b`. The controller changes were checked
on `codex/two-sided-agv-controller`. These are development working-tree probes,
not formal experiments, a milestone promotion, or evidence of optimality or
deadlock freedom on arbitrary maps. Generated evidence remains local.

The simulator physics and current V3.2 public contract are unchanged. Pickup
intentions are nonexclusive; historical matching author names compile to
`first_arrival`. The policy uses only public observations and semantic actions.

## Problems addressed

- The historical dispatcher admitted one inbound car per source rather than
  per physical port, wasting a station's second independently usable side.
- The new target-only dispatcher requests only selected empty AGVs. A global
  assignment to an uncalled vehicle could leave the actual requesting car
  without useful work. Ranking now uses the requesting vehicle and soft
  source/port queue pressure, not inventory reservations.
- Empty target intentions were treated like useful transport. Standby vehicles
  now park off ports, yield to useful traffic, and resume when work appears.
  Only a bounded set of spare cars travels to trigger retargeting for uncovered
  ready sources.
- A vehicle could enter a transit port while its forward exit stayed occupied,
  then be forced back by the core's port-clearance mask. Planning clears empty
  blockers before entry and checks the joint next-state exit. That admission
  check does not forbid legal waiting/clearance on an already occupied port.
- Shared approaches are protected without serializing opposite station sides
  whose exits are independent. Loaded vehicles retain available destinations;
  final UNKNOWN work is sent to inspection before delivery.

## Recorded probe results

Each case uses 12 demands, four operations, unit nominal processing durations,
seed 202, traffic admission, no explicit active-fleet cap, and a 400-tick limit.
These settings are not necessarily the user's previous replay settings, so the
ticks are not a controlled before/after performance comparison.

| Bundled map | Fleet | Completion tick | PASS shipments | AGVs with pickup/drop work | Sustained reversal flags | Longer-loop flags | Rejected moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Template 7, small | 8 | 269 | 12 | 8 | 0 | 0 | 0 |
| Template 8, medium | 16 | 201 | 12 | 16 | 0 | 0 | 0 |
| Template 9, large | 32 | 183 | 12 | 26 | 0 | 0 | 0 |

Every recorded final state contains 12 PASS jobs at `buffer_002` and one FAIL
attempt at a scrap bin. Shipment counts alone would not establish quality under
the current blind-shipment semantics; final job quality/location was checked
separately. All three replay audits passed with `trajectory_complete: true` and
action contract `smartsom.production-actions/v3.2`.

Small-map seed 101 completed at tick 196 and seed 303 at tick 204. Both had 12
completions, all eight vehicles doing pickup/drop work, and zero reversal,
longer-loop, or rejection flags. They were diagnostic runs without recording;
no independent final-quality claim is made for those two reports.

Both physical sides of each small-map inspection station were used:

| Station | Port | Pickups | Drops |
| --- | --- | ---: | ---: |
| inspection_station_001 | port_017 | 3 | 6 |
| inspection_station_001 | port_023 | 6 | 4 |
| inspection_station_002 | port_018 | 5 | 7 |
| inspection_station_002 | port_024 | 9 | 7 |

Using both sides over a run does not itself establish simultaneous service.
Independent-side concurrency is covered by a focused planner regression.

## Evidence and reproduction

Local evidence paths, relative to the repository root:

- `runs/diagnostics/two_sided_acceptance.json`
- `runs/diagnostics/two_sided_acceptance_small_seeds.json`
- `runs/visual_replays/two_sided_2026_10_07_acceptance/template_007_seed_202_active_0`
- `runs/visual_replays/two_sided_2026_10_07_acceptance/template_008_seed_202_active_0`
- `runs/visual_replays/two_sided_2026_10_07_acceptance/template_009_seed_202_active_0`

The reports preserve the source commit, dirty-path list, and code hashes. The
recorded controller hashes are:

| File | SHA-256 |
| --- | --- |
| `src/smartsom/algorithms/agv_planner.py` | `96d359bb553160b90fd3f0ccb67ad8e39fd44cc8bb2e4ba22bd33665f0e49ced` |
| `src/smartsom/algorithms/agv_dispatcher.py` | `49ebdbe7143a7e264d71b706065ae8e28af561af9b57355b49434448886a649d` |
| `src/smartsom/algorithms/production_rules.py` | `87b06544d54e330f2495521ffa6d6de9ff66baa574c6098e5cdd5cb7ea619883` |
| `scripts/validation/rule_controller_probe.py` | `e17fd4b7949f48c7b3fb7281ba4ba048d967da7ee8c6786e731860cae143ae0e` |

From the repository root in PowerShell, use a new recording root rather than
overwriting existing evidence:

```powershell
uv run python scripts/validation/rule_controller_probe.py --templates 7 8 9 --jobs 12 --seeds 202 --active 0 --admission traffic --limit 400 --output runs/diagnostics/two_sided_repeat.json --record-root runs/visual_replays/two_sided_repeat
uv run smartsom audit runs/visual_replays/two_sided_repeat/template_007_seed_202_active_0
uv run smartsom playback runs/visual_replays/two_sided_repeat/template_007_seed_202_active_0
```

`peak_active` counts intentions, not useful work. Use `services_by_agv` and
`working_ticks_by_agv` to assess activity. Sustained reversals require the same
task and consecutive physical ticks; stationary gaps reset the streak. The
unchanged longer-loop diagnostic checks 16 moves of one task confined to four
cells or fewer, including moves separated by waiting. Neither diagnostic is a
complete cycle/deadlock detector.

## Verification boundaries

Focused controller, physical-contract, rule registry, scale, configuration and
workflow tests finished with **123 passed, 4 skipped**. Ruff and formatting
passed, as did the four-file input check and all three replay audits. The
combined test command was:

```powershell
uv run pytest tests/unit/test_agv_dispatcher.py tests/unit/test_agv_planner.py tests/unit/test_rule_controller_probe.py tests/unit/test_rule_registry.py tests/unit/test_grid_composition.py tests/unit/test_composable_physics.py tests/unit/test_composable_config.py tests/unit/test_composable_workflow.py tests/integration/test_rule_controller_scale.py -q
```

New regressions cover independent sides, shared exits,
standby yielding/resumption, transit entry versus already-occupied-port waiting,
four-vehicle rotation without swaps, and diagnostic tick gaps.

A broader Windows unit run during integration reported 59 failures, 1996
passes, 189 skips and 10 deselections. Failures included path separators,
POSIX-specific process/locking assumptions, symlink privileges, encoding,
terminal output and byte-hash expectations. They were not all independently
classified against a clean baseline, and that run predates the final controller
iteration. The full repository quality gate is therefore not claimed clean.
Optional training/solver environments and general-topology liveness were not
validated by this checkpoint.

## Controller-only publication

The controller was prepared for publication on
`codex/two-sided-agv-controller-review`, based directly on GitHub `main` at
`bf7bf20`. Its source/configuration changes match the tested local checkpoint
`783255c` after Git text normalization. Local replay-history commits and their
unrelated architecture documentation were excluded; generated recordings were
not included.

In the isolated publication checkout, focused dispatcher/planner/diagnostic
tests passed **28 tests**. Rule identity, composition, physical behavior,
configuration, four-file authoring and all three scale regressions passed
**161 tests, with 4 skips**. Ruff, formatting, the four-file input check and
staged whitespace checks also passed. Formatting normalized mixed Windows line
endings without changing source logic. Raw byte digests can therefore differ
from the earlier recording table above; the original recordings and their
source identities remain historical local evidence, not reruns from this
publication commit. The broader Windows gate limitation remains as described
above. Publication is for review, not a merge into `main` or a release.
