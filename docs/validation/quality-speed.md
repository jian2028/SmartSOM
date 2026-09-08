# Quality-speed modes: item-10 acceptance

The item adds configurable shared/per-machine quality capabilities, exact integer
speed scaling, independent operation draws, final inspection and online fixed-label
baselines. No rework, early scrap, inspection resource/duration, quality objective,
learning, quality CP or batch interface is included. The normative contract is
[ADR 0008](../decisions/0008-quality-speed-and-final-inspection.md).

## Use and input ownership

```bash
uv run smartsom validate configs/runs/quality_m0.yaml
uv run smartsom run configs/runs/quality_m0.yaml       # 24, passing rate 1
uv run smartsom run configs/runs/quality_m1.yaml       # 20, passing rate 0
uv run smartsom run configs/runs/quality_m2.yaml       # 16, passing rate 0
uv run smartsom run configs/runs/quality_generated.yaml
uv run smartsom run configs/runs/quality_hidden.yaml
uv run smartsom run configs/runs/quality_machine.yaml
uv run smartsom run configs/runs/quality_combined.yaml
```

Factory example (inside `factory`):

```yaml
quality_speed:
  default_modes:
    - {quality_mode_id: M0, time_scale: '1.2', error_rate: '0.01'}
    - {quality_mode_id: M1, time_scale: '1.0', error_rate: '0.018'}
    - {quality_mode_id: M2, time_scale: '0.8', error_rate: '0.03'}
  machine_modes:
    - machine_id: M2
      modes:
        - {quality_mode_id: M0, time_scale: '1.4', error_rate: '0.005'}
        - {quality_mode_id: M1, time_scale: '1.0', error_rate: '0.015'}
        - {quality_mode_id: M2, time_scale: '0.7', error_rate: '0.04'}
```

M2's table replaces the whole default table. Different labels/counts are allowed,
but a fixed-label algorithm requires that label on every candidate base mode.
Missing a used machine's table is an error; no operation-level override exists.

The scenario uses `quality: {kind: independent_operation_v1}` to generate draws,
or `quality: {kind: fixed, path: PATH}` to import `realized_quality.json`.
`probability_visibility: hidden` can be added to either; public is the default.
The selected algorithm uses `parameters: {quality_mode: M0}` for fixed M0, or
omits/nulls that parameter for free selection. Root seed remains in `run.yaml`.

Python construction uses immutable `QualityMode`, `QualitySpeedSpec`,
`QualityDraw`, and `QualityDrawPlan` from `smartsom.domain.quality`.
`smartsom.modules.quality.prepare_quality(factory, workload, draws,
processing_times=base_upt)` returns the `QualityPlan` supplied as `quality=` to
Simulator, action replay or schedule replay. These entry points also accept
`quality_probability_visibility="public" | "hidden"`. `quality_mode_id(base, label)`
constructs script identities without depending on catalog positions.

Base `realized_instance.json` and `realized_processing_times.json` are unchanged.
`realized_quality.json` stores the independently reusable operation draws;
`effective_modes.json` records the full base-to-execution mapping, exact parameters
and nominal/actual execution times. Catalog and draw hashes are separate from
base workload/UPT hashes and from raw source-byte hashes. Changing an imported
run's root seed does not regenerate any fixed input. Changing the algorithm does
not regenerate the world. Probability hiding changes only the projection.

Evidence records `quality_check` at processing completion and `inspection` at
real output, including output reached in the automatic unload closure. Decisions
only receive final job inspection status after output. Privileged files include
latent inputs and actual operation outcomes and are not the policy observation.
A finished simulation result has `quality`; quality-off results have `None`.
Successful summaries add counts, equal-job passing rate and inspected outcomes.
Progress reports inspected/passed/defective counts, not unpublished defects.
Failed summaries retain null makespan and do not report final passing rate.

## Independent deterministic references

`data/reference/quality/workload.json` is a manually specified two-operation route
M1/10 -> M2/10. Fixed draws are u1=1/64 and u2=1/2. The registered expectation is:

| Policy | A1 | A2 | Makespan | Job passes |
|---|---|---|---:|---|
| SPT-M0 | [0,12) | [12,24) | 24 | yes |
| SPT-M1 | [0,10) | [10,20) | 20 | no |
| SPT-M2 | [0,8) | [8,16) | 16 | no |

For M1/M2, A1's error is followed by A2's successful operation, but the job remains
nonconforming. Its full route still executes. Probability endpoints and strict
threshold equality are separate tests using exactly representable draws.

The UPT/outage case fixes base actual durations 12 and 3. M2 produces net work 10
and 2. M1 outages [2,4), [6,8) yield operation intervals [0,14), [14,16), with
exactly one quality check per operation, not one per processing segment.

The direct zero-buffer case completes operations at 10 and 25 while explicitly
holding/moving the job; output at 30 is the only inspection time. Tests also
cover completion coincident with breakdown and automatic loaded-vehicle unload.

The fixed PDF source is Draft4, page 4, SHA-256
`72023b1162f1b6fc43731191c78cdb45aa4b535e5b11d31b4026da79660bb66f`.
It establishes example parameters and sticky defect semantics. We do not claim
old full-factory numerical reproduction or paper authority for the newly agreed
RNG coupling, final-only observation and rounding choices.

## Preregistered probability acceptance

Before implementation, root seeds **10,000,000 through 10,009,999**, five-operation
routes and absolute tolerance **0.025** were fixed. The checked reference resides
in `data/reference/quality/reference.json`; it is not inferred from results.
Each of four routes uses 10,000 complete simulator micro-episodes with independent
operation draws and fixed semantic actions. The expectation is the product of
per-operation success probabilities. The seed range/tolerance are never adjusted
to pass a run.

| Five-operation route | Analytic passing probability | Observed | Absolute error |
|---|---:|---:|---:|
| M0 × 5 | 0.9509900499 | 0.9512 | 0.000210 |
| M1 × 5 | 0.9131822030 | 0.9109 | 0.002282 |
| M2 × 5 | 0.8587340257 | 0.8536 | 0.005134 |
| M0,M1,M2,M1,M0 | 0.9167799338 | 0.9141 | 0.002680 |

These are bounded engineering checks against a specified probability model, not
research performance results or proof of realistic manufacturing quality.
No external RNG or simulator is treated as a complete numerical oracle.

The small generation golden uses root 42, quality seed `1400014650531944465`,
and A1/A2 draws `2092992623612522` / `7285602162975817`. The canonical draw digest
is `3c16f4f62d3df9e029dcf15a616816fa35a9700eaa8158dc007025b2b513e205`.

## Composition, compatibility and failure checks

Focused tests cover strict configuration/types/coverage, whole-table override,
fixed-label rejection, atomic invalid dispatch, immutable snapshots and plans,
shared cross-mode draws, independent operation identity, global RNG isolation,
input order/hash seed/cwd independence, exported input reuse and no double scaling.
Hidden-probability and changed-latent-input comparisons verify preinspection
observations. Provider, script and evidence-write failures preserve available
inputs and trace without terminal quality success metrics.

With quality enabled, all 32 JA/MB/UPT/AGV/buffer switch combinations run through
the core; the 16 with JA also run the second existing trigger. Tests include a
forced defective operation and both free/fixed-label SPT. The configured harness
uses 48 combinations × three providers/presets (SPT, SPT-M0, first-feasible), seven
examples and one exported-input reimport: **152 run_one attempts**. Every attempt
checks direct policy execution, exact action replay and schedule replay.
Action replay compares complete results. Timetable replay compares every schedule,
quality fact and final metric; it need not reconstruct redundant waiting actions
that were not part of a timetable's history. v1/v2 schedule shapes are unchanged.

A preimplementation capture of **28 legacy configured cases** verifies exact
workload, named-seed, schedule, complete-trace, observation and makespan equality
with quality off. Input goldens and optional-dependency import checks remain in
the full suite. Zero error probability while quality is enabled is intentionally
not full-trace equivalent to disabling the module.

## Commands and evidence

```bash
uv run --locked --no-sync pytest -q tests/unit/test_quality.py tests/unit/test_quality_inputs.py
uv run --locked --no-sync python scripts/validate_quality.py --output-dir artifacts/item10/acceptance
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv lock --check
uv pip check
uv run --locked --no-sync pytest -q
SMARTSOM_REQUIRE_CP=1 uv run --locked --no-sync pytest -q tests/integration/test_cp_sat.py
git diff --check
```

The base environment must pass without CP. The pinned CP environment must actually
run ft06=55, official FJSP=6 and Mk01=40; skips cannot satisfy that gate.
Local `artifacts/item10/precommit/acceptance.json` records initial configured and
probability acceptance. `artifacts/item10/final-checks.json` records final checks
and the post-commit acceptance path with integrated source identity. Generated
runs and aggregate acceptance files are not committed. `--skip-statistics` is
available only for configured smoke and explicitly marks acceptance incomplete.
