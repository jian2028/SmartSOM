# Shared holding-buffer acceptance

```sh
uv run smartsom validate configs/runs/holding_hand.yaml
uv run smartsom run configs/runs/holding_hand.yaml
uv run pytest -q tests/unit/test_holding.py tests/unit/test_holding_inputs.py
```

The configured hand case is independent of the frozen large IDETC inputs:

| Movement/work | Start | End |
|---|---:|---:|
| Input → M1 | 0 | 1 |
| J1 net work | 1 | 3 |
| M1 → holding | 3 | 4 |
| Deliberate idle | 4 | 6 |
| Holding → M2 | 6 | 7 |
| J2 net work | 7 | 10 |
| M2 → output | 10 | 11 |

Both processing durations and all physical travel/idle intervals are checked.
Expected makespan is **11**, with one operation start/completion each and exact
action/full-schedule replay. Expectations were registered before runtime changes
in `artifacts/idetc/holding-checks/preregistered.json`.

Independent capacity-one competition: A books at tick 1 and arrives at 3; B books
without reservation at 1 and arrives at 2. B unloads only after A's actual pickup,
at 3 (or 5 with a distant pickup vehicle). Earlier arrival cannot borrow A's slot.
Another case forces the current SPT fallback from a completed-blocked machine when
the next prebuffer is full, then resumes normal delivery/processing: makespan 12.
Zero/one-capacity physical deadlocks are explicitly checked and do not recover.

Tests cover direct blocked pickup, delayed source release, paused/input/prebuffer
parking refusal, invalid references/capacities, immutable views, old disabled trace
encoding, reordered/hash-seed/cwd inputs, malformed v2 schedules, configuration
snapshot reload and writer failure evidence. The 32 JA/MB/UPT/quality/machine-buffer
combinations actually travel through holding; JA additionally uses both decision
triggers. Fixed future repair changes do not change current holding observations.

`data/reference/holding/source.json` records the frozen old feature's source hash.
We do not run the old RL/Gym stack or assert equivalent old algorithms, random
streams, blocking behavior or terminal-clock semantics. IDETC's 60 runs remain a
separate postcommit integration gate.

Fresh macOS precommit gate: 65 focused holding tests; base 823 passed / 12 CP
skips; locked CP 835 passed / zero skips. Ruff, format, lock, dependency and diff
checks passed. Actual logs are under `artifacts/idetc/holding-checks/`; Ubuntu CI
is configured but has not been executed in this local task.
