# Week2 item 7 — Machine breakdown and automatic repair resumption

The engine supports finite machine outages, original-machine pause/resume,
independent seeded generation, strict fixed import, online policies and exact
replay. It composes with arrivals, actual processing times and multiple modes.
See [ADR 0005](../decisions/0005-machine-outages-and-processing-progress.md) for
state ownership, visibility, event phases and the unchanged decision triggers.

## Public interface and configuration

```python
from smartsom.domain import MachineOutage, MachineOutagePlan
from smartsom.engine import Simulator, replay, replay_schedule

events = MachineOutagePlan((MachineOutage("M1", 2, 4),))
simulator = Simulator(factory, workload, machine_events=events)
```

All three entries accept `machine_events=None` by default alongside their existing
arrival/processing options. Replay requires the same actual inputs. Outage times
must be strict integers with `0 <= start < end`; unknown machines fail validation.
Unions merge overlapping, duplicate and adjacent intervals, independently of
source order. Empty fixed plans preserve the original schedule/action/trace.

```yaml
machine_events:
  kind: fixed
  path: ../../data/machine_events/hand.json
```

```yaml
machine_events:
  kind: exponential_uptime_v1
  profile:
    generation_until_tick: 100
    machines:
      - machine_id: M1
        mean_uptime_ticks: 20
        repair_ticks: {min: 2, max: 5}
      - machine_id: M2
        mean_uptime_ticks: 40
        repair_ticks: {min: 1, max: 3}
```

Profile rows must be nonempty, unique by machine ID and refer to known machines.
The cutoff is a positive integer. Means are positive finite numbers, normalized
to binary64; booleans and strings are rejected. Repair bounds are positive strict
integers with min <= max. No implicit reliability defaults or extra seed authority
are introduced. Unknown fields and duplicate JSON/YAML keys fail strict loading.

## Exact v1 generation and evidence

1. Take the existing derived `machine_events` unsigned 64-bit seed.
2. Encode `["smartsom.machine-events/v1", seed, machine_id]` as compact UTF-8 JSON
   with Unicode preserved. Interpret its complete SHA-256 as a big-endian integer
   and create a local `random.Random` for this machine.
3. Starting from `repaired_at=0`, draw `u=random()`, then `x=-log1p(-u)`.
   Compute `uptime=max(1, ceil(x*mean_uptime_ticks))`. The multiplication uses exact
   fractions of the two binary64 values, avoiding overflow and fixing the ceiling.
   This is upward-discretized exponential uptime, not an assertion that the
   integer samples have exactly the configured continuous mean.
4. If `repaired_at+uptime` is at/after the cutoff, stop this machine. Otherwise
   draw the repair duration with inclusive `randint(min,max)`, retain the full
   interval, and continue from its repair end. Idle time counts as uptime.

The generator consumes randomness even if its first draw lies outside the window.
Fixed imports and disabled inputs consume no machine-event randomness. Extending
the cutoff preserves the earlier generated prefix; editing another machine does
not shift this one's stream. All other named seeds and samplers are unchanged.

The unit golden uses effective seed 42, cutoff 20, M1 mean 3 / repair 1..3, M2
mean 5 / repair 2..4. It yields M1 `[1,4), [8,10), [11,13), [14,15), [16,19)`
and M2 `[6,9), [15,19)`. The expected values were also calculated independently
using SHA-256 and `Random.expovariate`; they are not regenerated as test expectations.

`realized_machine_events.json` uses schema `smartsom.machine-events/v1`, containing
`machine_events: {outages: [...]}`, optional `content_sha256` and optional provenance.
Generated provenance records generator/version, normalized profile, profile digest
and effective seed. The version specifies the per-machine seed recipe above.
Import verifies declared digests and actual machine references; it does not rerun
the sampler or authenticate a file's historical-generation claims.

The manifest separately records machine-event content/provenance and original
source-byte digests. Arrivals keep their existing file; actual processing times
keep theirs. `observations.jsonl` records only delivered policy views, including
availability and completed net processing ticks. The unified trace adds actual
machine events and pause/resume, without repair forecasts or another dispatch.
Failed attempts preserve partial evidence and never report a successful makespan.

## Independent reference checks

All single-operation cases start at zero and require five processing ticks:

| Outages | Active processing segments | Completion |
|---|---|---:|
| [2,4) | [0,2), [4,7) | 7 |
| [1,3), [4,6) | [0,1), [3,4), [6,9) | 9 |
| [5,8) | [0,5) | 5 |

The [PyJobShop breaks example](https://pyjobshop.org/stable/examples/breaks.html)
adds two three-tick tasks, a machine break `[4,5)`, and `allow_breaks=True`.
With starts fixed at 0 and 3, completion is 3 and 7. The reference harness fixes
starts to compare physical execution, not to measure online policy quality.
PyJobShop 0.0.9 / OR-Tools 9.12.4544 actually return optimal objective/bound
**7, 9, 5, 7**, with task spans and reconstructed active segments matching the
independent fixtures and core replay. Four additional CP tests require this solve.

DynaSchedBench source commit `08975bf4a0473c5dff9177393bc6743db9ddc946` provides
an additional single-machine, single-operation, speed-one check. The controller
explicitly redispatches job A on M1 immediately after repair. It obtains active
segments and completions **7, 9, 5**. Job A maps to SmartSOM's J0/O0 in this fixture.
This is not action-trace or general simulator equivalence: DynaSchedBench returns
interrupted work to waiting, whereas SmartSOM resumes internally.

Its raw `get_gantt()` retains old projected ends after interruption. Accordingly,
we preserve that raw output and compare the actual machine `schedule_segments`
and final job completion. We do not repair or monkeypatch the external simulator.
Its unused MOO/hybrid calibrators warn when pymoo is absent; neither is used.

The small fixed inputs, raw numerical outputs, source hash and dependency versions
are under [data/reference/machine_events](../../data/reference/machine_events).
The external source remains outside SmartSOM. The separate validation requirements
pin its import dependencies and never become base or CP project dependencies.

## Commands and verification

```bash
uv run smartsom validate configs/runs/machine_events_generated.yaml
uv run smartsom run configs/runs/machine_events_fixed.yaml       # 22
uv run smartsom run configs/runs/machine_events_generated.yaml   # 26
uv run smartsom run configs/runs/machine_events_arrivals_dispatch.yaml  # 22
uv run smartsom run configs/runs/machine_events_arrivals_event.yaml     # 22
uv run --no-sync python scripts/validate_machine_events.py --output artifacts/item7/reference.json
uv run --no-sync python scripts/validate_machine_events.py --cp --output artifacts/item7/cp-reference.json
uv run --no-sync --with-requirements scripts/validation/dynaschedbench-requirements.txt python scripts/validate_machine_events.py --dsbx-root /path/to/pinned/dynaschedbench --output artifacts/item7/dsbx-reference.json
```

The last command uses a temporary dependency environment. DynaSchedBench is
read from the explicit source checkout, whose clean status and pinned commit are
checked before import. It requires no LLM credentials, paid service or calibration.

Focused tests cover interval unions, exact active segments, completion-first,
tick-zero/idle failures, simultaneous phases, cancelled events, paused occupancy,
mode rejection, automatic continuation, private future inputs and completed
net-work disclosure. Both arrival triggers and processing uncertainty compose;
SPT uses available nominal modes. Input tests cover strict validation, immutable
resolution, seed/provider-independent reimport, hash seed/cwd, input deletion
after resolution, and partial evidence after policy/write failure.

Before/after comparison of all **17** pre-existing online examples preserves
factory/workload/arrival/processing inputs, every named seed, full result, schedule,
actions and complete trace against baseline commit `01809e8`. Snapshot additions
are deliberate; old trace records themselves do not change.

Run focused checks before the complete base and locked CP suites, dependency
checks, Ruff, formatting and `git diff --check`. CP also retains ft06=55,
official FJSP=6 and Mk01=40. Development and post-commit reports live under ignored
`artifacts/item7/`; only small intentional reference fixtures are committed.
Post-commit reports must identify the new main commit and rerun configured fixed,
generated, imported, combined and reference paths. Dirty governance/paper work
remains explicitly reported by manifests and is not included in this change.

Pre-commit verification on 2026-09-07: **70 focused machine-event tests passed**;
the independent base environment returned **520 passed, 11 optional CP skips**,
and the locked CP environment returned **531 passed**, including all 11 actual
solver/reference checks. Both environments passed dependency consistency.
The pinned DynaSchedBench validation was also rerun against the fixed raw outputs.

These are bounded engineering checks, not large-scale performance evidence.
Full trace remains in memory. No dynamic CP, debug logging, batch, repair workers
or logistics is implemented. Before logistics, reconcile the Week2 AGV-first list
with the roadmap's buffer-first sequence and establish resource ownership.
