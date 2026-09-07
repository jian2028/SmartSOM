# Static FJSP acceptance — Week2 item 4

This slice adds multi-mode execution, import and generation to the same static
engine. It retains serial job chains, initially available jobs, positive integer
ticks, non-preemptive processing and capacity-one machines. Runtime/difficulty
classification, quality/speed interpretations, dynamic modules and batch remain
out of scope. This record establishes engineering behavior, not a research claim.

## Mode semantics and hand cases

A mode is identified by `(operation_id, processing_mode_id)`. All modes are
preserved, including same-machine alternatives with different durations or equal
durations and distinct IDs. Selecting one starts the operation once; alternatives
are no longer candidates. Immutable operation snapshots retain the chosen mode.
Events, machine occupancy, intervals and invariants use that mode's duration.

The hand instance is `data/instances/fjsp_hand.json`:

| Operation | Alternatives | Predecessor |
| --- | --- | --- |
| A1 | slow: M1/4; fast: M2/1 | none |
| A2 | standard: M1/2 | A1 |
| B1 | standard: M2/2 | none |
| B2 | standard: M1/1 | B1 |

| Path | Actual intervals | Makespan |
| --- | --- | --- |
| fast | A1/M2 [0,1), A2/M1 [1,3), B1/M2 [1,3), B2/M1 [3,4) | 4 |
| slow | A1/M1 [0,4), B1/M2 [0,2), B2/M1 [4,5), A2/M1 [5,7) | 7 |

`configs/runs/fjsp_fast.yaml` and `fjsp_slow.yaml` submit literal semantic scripts.
Their complete results match code-built inputs and exact schedule replay.
Additional tests preserve same-machine equal-duration IDs, explicitly dispatch
the slower mode, reject another-mode redispatch before mutation, and check busy
machines, incomplete predecessors and immutable snapshots. SPT selects the
minimum `(nominal_ticks, operation_id, processing_mode_id)` among current legal
candidates. It does not wait for a busy faster alternative.

Input reordering preserves fixed-action and fixed-schedule results. Hash-seed and
working-directory checks compare canonical instance/result serialization. Existing
single-mode full trace expectations are unchanged. SPT can submit simultaneous
starts in a different order from canonical schedule replay; schedule equality
and fixed-action trace equality are the appropriate checks for those two paths.

## Traditional `.fjs` import

```bash
uv run smartsom import-fjs data/reference/mk01/Mk01.fjs \
  --instance-id mk01 --output-dir artifacts/imported-mk01
```

The importer accepts UTF-8 text. The first nonempty line contains positive integer
job/machine counts and optionally a positive finite average-flexibility token.
Each subsequent nonempty line is one complete job: operation count, then for each
operation its alternative count and `(machine, duration)` pairs. Machine numbers
are in `1..M`; counts and durations use decimal integer tokens. Missing/trailing
tokens, wrong line counts, extended formats, comments and invalid numbers fail.
The optional average is retained as its original string and is not checked
against the realized average (Mk01's header is `10 6 2`).

`import_fjs(path, instance_id=ID)` returns immutable factory, workload and import
provenance. IDs are `ID`, `ID/job_N`, `ID/job_N/op_K`, and `M1..M`. Each mode ID is
`m{machine}_t{duration}_v{occurrence}`, with occurrence starting at 1 for each
identical pair in that operation. Alternatives are canonically sorted; duplicate
pairs remain separate modes. Reordering alternatives changes raw bytes, but not
domain content. The full file is read once for parsing and SHA-256.

The command creates `factory.yaml` and `workload.json` only after validation,
refusing even an empty existing output directory. Parsing errors return exit 2;
output errors return exit 1. The exported files use the existing schemas. A
scenario in that output directory can use:

```yaml
schema: smartsom.scenario/v1
factory: factory.yaml
workload: {kind: instance, path: workload.json}
visibility: full_static
```

Reference this scenario from an ordinary run configuration. `.fjs` is an ingress
format, not a new scenario source kind. Import provenance records importer
`fjs_v1`, version `1`, raw digest, instance ID and header. Manifest fields
`import_provenance` and `generation_provenance` preserve their separate meanings.
Run seed/provider changes never regenerate an imported instance.

## Independent FJSP generation

`configs/workloads/static_fjsp.yaml` uses `smartsom.workload-profile/v1` with:

```yaml
generator: static_fjsp_v1
profile:
  order_count: 1
  jobs_per_order: 2
  operations_per_job: {min: 2, max: 4}
  eligible_machines_per_operation: {min: 1, max: 3}
  nominal_ticks: {min: 1, max: 5}
```

All counts/range endpoints are strict positive integers; min cannot exceed max.
Only the eligible-machine maximum is bounded by machine count. Operations may
revisit machines and operation counts may exceed machine count.

The version-1 recipe uses a local `random.Random(workload_seed)`. Visit numeric
order/job indices, draw each job's operation count, then per operation draw its
candidate count, sample candidates without replacement from sorted machine IDs,
and draw durations in selected machine-ID order. Every range is closed and uses
integer uniform sampling, including fixed bounds. One mode per selected machine
has ID `machine/{machine_id}`. Nominal times remain fixed during execution.

Root seed 42 produces workload seed `6941565647359864201`. With the committed
three-machine profile, the six operation candidate lists are:

```text
job_1/op_1: M1/5 M2/1 M3/1
job_1/op_2: M2/3
job_1/op_3: M1/3 M2/3
job_2/op_1: M1/1 M2/2 M3/3
job_2/op_2: M1/5 M3/5
job_2/op_3: M1/1 M2/1 M3/1
```

The canonical workload digest is
`35da0e9853412307a15e33162d7fd8eb4070ceb62aca9b8a9f21bcce4afc9db2`.
The original JSP golden remains
`c039e0307dc2c71cc3b29d6e1ee3ad466dd7055b44422a2fd254d0678d0ee7a5`.
Generator/profile matching is strict. No cross-generator same-seed equivalence
is promised; neither generator changes global RNG state.

## External reference snapshots

| Instance | Jobs / machines / operations / modes | CP objective / bound / engine |
| --- | --- | --- |
| PyJobShop official small FJSP | 3 / 3 / 9 / 27 | 6 / 6 / 6 |
| Brandimarte Mk01 | 10 / 6 / 55 / 115 | 40 / 40 / 40 |
| ft06 regression | 6 / 6 / 36 / 36 | 55 / 55 / 55 |

All three real adapter checks require `OPTIMAL`. CP remains PyJobShop 0.0.9 /
OR-Tools 9.12.4544, one worker, default 60 seconds. The root-42 effective solver
seed is `9725273414194853204`, mapped to backend seed `1017279828`. The horizon
guard sums each operation's longest candidate duration. Every external global
mode index must map to its owning operation; all modes are represented without
pruning. No other solver constraint or parameter interface is added.

- The [official PyJobShop example](https://pyjobshop.org/stable/examples/flexible_job_shop.html)
  supplies the 27 duration/machine pairs and reports optimum 6. The pinned data
  cell comes from notebook commit `0510b1ba022641fd6590922ae865550fd573fe43`.
  `data/reference/pyjobshop_fjsp/` retains the extracted cell, literal data,
  notebook byte digest, conversion and source license.
- Mk01 raw bytes come from [PyJobShop/Instances](https://github.com/PyJobShop/Instances/blob/18bc8d999b937d882acade567e3ac9b9cf6dc1c5/FJSP/Brandimarte/Mk01.fjs),
  commit `18bc8d999b937d882acade567e3ac9b9cf6dc1c5`; SHA-256 is
  `c0bed4ae79833ab73adcc653dd85c7cdfd66bb8c6a9744ae781719d191e6e14d`.
  The raw file retains its final blank line; `.gitattributes` disables only the
  blank-at-EOF check and text conversion for this exact source fixture.
  [ScheduleOpt's FJSPLib table](https://scheduleopt.github.io/benchmarks/fjsplib/)
  independently records bounds 40/40. That bound observation is not a timetable.

The new fixed reference schedules were generated separately using the pinned
PyJobShop API, before using SmartSOM's adapter: Mk01 used `Model.from_data(read())`;
the small example used direct numeric-index modeling of the literal data. Both
references use the settings above. They are newly generated reference fixtures,
not published source schedules. `direct_solver_result.json` retains external
task/mode/resource indices and intervals; `sources.json` binds snapshot hashes,
semantic schedule digest and workload digest. Tests decode the original data
independently, compare every mode, and verify the reference intervals through
the engine. Source MIT notices accompany the extracted data.

Actual adapter solutions independently undergo exact replay and may differ from
the fixed optimal reference. Solver output is saved before replay; invalid mode,
machine or duration output retains `solver_result.json` and failure evidence,
without a successful makespan. Existing no-incumbent, feasible-but-unproven,
provider exception and post-replay verification failure tests remain in place.

## Verification and source identity

Run focused FJSP tests first, then both locked environments:

```bash
uv run --no-sync pytest -q tests/unit/test_fjsp_engine.py tests/unit/test_fjsp_inputs.py tests/unit/test_fjsp_solvers.py
uv sync --locked
uv run --no-sync pytest -q
uv pip check
uv sync --locked --extra cp
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
uv pip check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
git diff --check
```

The base environment must have no PyJobShop, OR-Tools or fjsplib installed; it
still exercises import, generation, SPT and fixed reference replay. CP CI must
execute the real solver tests, including reordered multi-mode inputs and
same-machine alternatives; the required job cannot treat a skip as acceptance.

The completed pre-commit checks on 2026-09-07 returned **284 passed, 7 skipped**
in the actual base environment and **291 passed** in the locked CP environment.
The seven base skips are precisely the optional CP integration tests; all seven
execute and pass in CP. Both installed dependency sets pass `uv pip check`.

Pre-commit tests verify the implementation. After the feature commit, rerun
import/export, generated-instance reuse, SPT, CP and fixed references, writing
source identity and actual outcomes under ignored `artifacts/week2-item4/`.
Those manifests identify the new integrated `main` commit and preserve any
unrelated dirty-file status. Generated run directories are not committed.
