---
name: evaluation-profile
description: Profile SmartSOM evaluation wall-time distribution and validate performance changes against a fixed-source baseline with identical cases and checkpoints. Use for evaluation bottlenecks, cache optimizations, or before/after speed comparisons.
---

# Evaluation profiling and optimization

Honor the requested scope: a profile-only request permits measurement helpers and local evidence, not production edits. Do not train unless requested. Read repository AGENTS.md and its triggered context before changes.

## Establish comparable work

Resolve the selected checkout/commit, implementation hash, frozen experiment inputs, case IDs and physical-world hashes, checkpoint hash, action seeds, policy selection, numerical threads, metrics level and stopping protocol. Inspect existing evidence first. Use an integrated main baseline for development comparisons; label feature-branch timings as engineering measurements, not formal research evidence.

For an incremental code review, use the stable pre-change revision as the immediate baseline and record its relation to main.

Compare identical workloads. Report completed/total jobs, termination reason, physical ticks/boundaries and decisions alongside time. A truncated run is not a faster completion of the same workload. A fixed-tick comparison and a completed-job comparison answer different questions. Preserve original runs; put generated evidence outside tracked source.

## Measure separately

The bundled [profile_case.py](scripts/profile_case.py) executes one frozen Base/SL learning evaluation in a fresh process. It supports baseline wall/CPU timing, nested phase timing, cProfile, and separate trajectory digests. Check `--help`; use the source checkout's locked uv environment. It requires a trusted native checkpoint and does not launch tracking, progress callbacks, training or panel orchestration. Report those exclusions. Unsupported policies/protocols require adapting and validating the helper, not silently changing the experiment.

Example, from the candidate worktree:

```sh
uv run --no-sync python .agents/skills/evaluation-profile/scripts/profile_case.py \
  --source /absolute/baseline-checkout --run /absolute/archived-trial \
  --checkpoint checkpoints/episode-00001000.pt --case 0 \
  --output /absolute/local-evidence --label baseline-case0 --mode baseline \
  --rebind-builtins
```

`--rebind-builtins` explicitly requests a development comparison with another implementation. The helper first validates the checkpoint against the archived algorithm identity and the archived source hash, then rebinds built-in component hashes in memory to the selected checkout. It records both identities. It does not bypass extension hashes, alter frozen inputs, or make the new code eligible to resume the old run. Without this option, existing component hashes remain enforced.

For serial paired runs, [compare_cases.py](scripts/compare_cases.py) accepts
`--baseline`, `--candidate`, repeatable `--run` / `--case`, `--repetitions`,
`--modes baseline phases trace` and a new `--output` directory. It alternates
execution order, compares complete reports except explicit wall-clock fields,
and fails on trajectory disagreement. Existing evidence is never overwritten.

For code-review regression checks, use at least three serial alternating repetitions. Compare wall-time medians: `(candidate / baseline - 1) * 100`. A value strictly above 5% on any representative case requires attention; do not average it away with improvements elsewhere. `compare_cases.py` records `performance_check` for uninstrumented timing only and emits an attention record without automatically failing the command. Its report/trace mismatch checks still fail. A run with fewer than three repetitions is explicitly preliminary. Investigate workload/source equality, host noise and the phase distribution; report the cause and disposition rather than treating 5% as automatic permission to change scope.

Run baseline and candidate timing serially, with several alternating repetitions and the same thread/host conditions. Include a full representative case for each affected policy/condition; expand if paths differ. Separate setup/import/checkpoint loading from episode execution. Record first-process/cold behavior separately from warmed repeated episodes if both are measured.

Ensure instrumentation preserves fast-path eligibility, RNG use and public callable identity. For the native single-request packer, time `_pack_one` underneath `learner.pack` so the profiler does not activate a custom-packer fallback; this excludes only the outer dispatch cost. Older implementations without that builder use the original pack wrapper. Record which helper version was used and reject phase distributions if hooks change the execution path.

Use `--mode phases` to report nonoverlapping shares: sum exclusive times, or a documented set of parent inclusive times with nested work excluded elsewhere. Never sum nested cumulative cProfile times. Use cProfile for call counts/callers and hotspot diagnosis; measure its overhead against baseline. Do not run timed cases alongside heavyweight tests or other benchmark cases.

## Preserve behavior

For cache proposals, state the key, lifetime, mutable dependencies, invalidation and ordering/numerical rules. SmartSOM can accept reservations, start processing and lock pickup/drop-off service multiple times at the same tick. Tick-only caches of these facts are unsafe. Preserve quota diagnostics, semantic IDs, tie-breaking, exact rational/rounding arithmetic, random draws and observation visibility. Native fast paths must retain supported extension behavior and configuration replacement semantics.

Try small, independently measurable changes. Compare a candidate against its actual main baseline, not merely a historical run. With `--mode trace`, compare request/action and physical-transition digests, counts, rewards and semantic report fields. Trace timings are instrumented and must not be used as speed evidence. Digests locate disagreement by category; investigate a mismatch with per-step capture. Matching selected traces is bounded evidence, not proof for all scenarios.

Run focused regression tests and the CI owners required by docs/ci.md. Clearly distinguish passed, failed and unexecuted platform gates. Do not weaken tests or resource guards to claim completion.

## Review and handoff

When invoked as the performance substep of an active research-code-review, return measurements and findings to that review; do not start a duplicate review loop.

For an implemented optimization outside an active review, use the colocated [research-code-review](../research-code-review/SKILL.md). Follow its independent engineering, scientific-semantics and testing reviews, defect verification and cross-testing; the implementing agent's self-check is not independent review. Reviewers inspect the exact source version read-only; the coordinator repairs confirmed defects only within the user's authorized development scope. A profile-only request does not start an implementation/review loop. Recheck invalidation, same-tick changes, ordering, floating-point reductions, retained tensors, external components and the measurement script itself. Record actionable findings before fixes; remeasure only affected results after a fix. If required independent review is unavailable or incomplete, report that gap rather than treating self-review or passing tests as a substitute.

Report the source pair, case coverage, timing scope, repetitions/median/range, wall-time shares, outcome/trace comparison, review findings and test gaps. Give local evidence links. Do not infer large speedups from hotspot percentages or compare different simulator semantics as equal work. A profile-only request ends with concrete proposals; implementation requires user authorization. No skill authorizes publishing, merging or contacting others.
