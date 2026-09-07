# Refactoring before Week2 item 7

This change preserves the item 1-6 behavior. It does not implement breakdowns,
batch execution, new logging controls, or bounded-memory simulation.

## Engine and incremental trace reads

Operation release times are indexed once in an immutable mapping. Invariants
reuse a single pending-calendar snapshot per check and the immutable set of
arrival events. All global checks and their execution frequency remain intact.
`Simulator.trace_since(cursor)` returns an immutable, non-consuming suffix;
the existing full trace and `SimulationResult` are unchanged.

The configured-run comparison against `f4bf22e4308b5e27a583eb1c6d6159a67be0f7bc`
covers all 17 online run presets. Normalized resolved inputs (excluding only the
operational output root), every named seed, complete simulation results,
observations, metrics, progress, realized inputs and semantic manifest fields
match exactly. Source metadata, attempt paths and resulting manifest hashes
naturally differ. CP is checked separately with actual solving and exact replay;
solver-generated schedules are not used as cross-version golden records.

### Local diagnostic

Five repetitions per size, eight machines, five operations per job, two eligible
machines per operation, nominal ticks in `[1, 5]`, `static_fjsp_v1` seed 101 and
SPT. Times include simulator construction and execution, excluding generation
and serialization. Peak allocations come from a separate `tracemalloc` pass,
not process RSS. This is a local engineering diagnostic, not a throughput claim
or CI timing threshold.

| Operations | Baseline median (s) | Indexed median (s) | Baseline / indexed peak bytes | Full observation bytes |
| --- | --- | --- | --- | --- |
| 50 | 0.008941 | 0.005393 | 93,424 / 88,344 | 983,392 |
| 100 | 0.039653 | 0.018569 | 230,012 / 232,892 | 3,909,804 |
| 200 | 0.209147 | 0.068798 | 472,972 / 473,524 | 15,572,023 |

For 200 operations, profiled calls fell from about 7.24 million to 1.57 million.
The repeated release scan is no longer a dominant cost. Full invariants remain
the main cost, and memory and observation volume are not materially reduced.
Observation sizes measure every public decision serialized as currently written
when observation recording is enabled; the timing case itself is static.

Local raw reports and the comparison driver are under the ignored
`artifacts/pre-item7-refactor/`. The complete existing tests additionally cover
reordered inputs, hash seeds, both arrival triggers, hidden information, module
disablement, semantic replay and retained failures.

The engine checkpoint passes 444 base-environment tests (seven optional CP
tests skipped only there), with the absence of PyJobShop, OR-Tools and fjsplib
verified. The locked CP environment runs all solver acceptances. Ruff, formatting,
dependency consistency and whitespace checks are also required before commit.
