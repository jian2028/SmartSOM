---
name: evaluation-profile
description: Profile SmartSOM RLlib Base/SL and Morris training and complete-episode sampling, or legacy evaluation. Use for runtime breakdowns, episode-cost estimates, and fixed-source performance comparisons.
---

# Evaluation profiling and optimization

Honor the requested scope: a profile-only request permits measurement helpers and local evidence, not production edits. Do not train unless requested. Read repository AGENTS.md and its triggered context before changes.

## Establish comparable work

Resolve the selected checkout/commit, implementation hash, frozen experiment inputs, case IDs and physical-world hashes, checkpoint hash, action seeds, policy selection, numerical threads, metrics level and stopping protocol. Inspect existing evidence first. Use an integrated main baseline for development comparisons; label feature-branch timings as engineering measurements, not formal research evidence.

For an incremental code review, use the stable pre-change revision as the immediate baseline and record its relation to main.

Compare identical workloads. Report completed/total jobs, termination reason, physical ticks/boundaries and decisions alongside time. A truncated run is not a faster completion of the same workload. A fixed-tick comparison and a completed-job comparison answer different questions. Preserve original runs; put generated evidence outside tracked source.

## Current RLlib profiling

Use [profile_rllib.py](scripts/profile_rllib.py) for the current Base/SL and
centralized Morris adapters. It uses the same public Algorithm/EnvRunner path
for both, with authored learner parameters, one local CPU learner and one local
EnvRunner. It does not change the production profile or launch Tune campaigns.
Training must be authorized; by default this helper performs bounded fresh
training, then samples one complete fixed-world episode with the resulting policy.
Use `--iterations 0` for sampling-only profiles of a freshly initialized policy,
or `--iterations N` to bound training by official iterations instead of waiting
for a complete training episode. Neither option changes learner parameters. For a large episode, use
`--sample-steps N` to compare the same finite EnvRunner step workload on baseline
and candidate; report actual physical/agent steps and in-flight metrics, not
whole-episode throughput. The helper hashes returned observations, actions and
rewards outside the sampling timer. That validation cost remains in end-to-end
wall time. Keep this option identical across paired runs.

```sh
PYTHONHASHSEED=0 uv run --no-sync python .agents/skills/evaluation-profile/scripts/profile_rllib.py \
  --source /absolute/checkout \
  --experiment /absolute/checkout/experiments/current/medium/sl-attention-shared-ppo/experiment.yaml \
  --output /absolute/new-evidence-directory --mode baseline --episodes 1 --seconds 900
```

Choose `morris-ppo/experiment.yaml` for Morris; the script is shared. For a machine
with a separate locked environment, use its interpreter through `uv run
--no-project --python /absolute/environment/bin/python` rather than syncing a live
experiment environment. Use fresh processes and new output directories. For a stable benchmark or regression claim, alternate methods serially for at
least three uninstrumented repetitions; then run `phases` separately. An initial
hotspot investigation may use one uninstrumented run and one phases run per
method, explicitly labeled preliminary rather than reporting a median or a
passed performance threshold. Do a tiny fixture validation before expensive cases. A deadline or
exception is a failed/incomplete measurement, never a faster episode. If an
authored training iteration is unexpectedly expensive, retain the interrupted
run, measure full sampling separately, and use a bounded `phases` training probe
to locate costs. `partial-training.json` is hotspot evidence only; it cannot
replace an uninstrumented completed-work measurement or establish training
seconds per episode. Interrupted sampling similarly writes `partial-rollout.json`
and available in-flight physical progress; do not label its wall time as a
completed episode.

The output freezes the world, scientific inputs, resolved AlgorithmConfig, source
and helper hashes. Startup, official training iterations, post-training rollout,
cleanup and driver peak RSS are separate. Training keeps official iteration
boundaries: it can sample part of the next episode. `seconds_per_ended_episode`
includes that extra work and counts truncations; inspect every terminal row before
using it. The rollout is inference plus environment time, not training throughput.
It uses a freshly trained, early policy, not a converged checkpoint. The single
world/condition and serial worker are diagnostic choices, not an estimate of the
whole multi-condition/multi-worker campaign. Tracking, checkpoint I/O, Tune and
fixed-panel orchestration are excluded explicitly.

`phases` wraps local sampling, inference, learner update, decision/observation,
traffic and physical stepping. Use exclusive phase seconds for disjoint totals;
inclusive times are nested and must not be summed. Compare each method's complete
non-timing results and iteration workload with its own uninstrumented run, and
report profiling overhead. Different methods need not have equal actions,
rewards or completion time. Their speed ratio is not an implementation regression
or the 5% migration gate. Preserve and report authored differences such as PPO
update epochs; do not silently equalize them.

## Legacy evaluation profiling

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
