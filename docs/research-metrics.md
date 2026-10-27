# Research metrics for Base and social-observation comparisons

The shared reservation rollout records `research_metrics` using schema
`smartsom.research-metrics/v1`. The native `smartsom run` reservation entrypoint
and the compatible four-file Base entrypoint use the same collector. This is an
engineering recording contract, not evidence that social learning improves Base.
Other simulator/algorithm entrypoints do not acquire this schema automatically.
A social-policy adapter using this rollout receives the same collector without
adding any observation to its actor or critic.

## Configuration and storage

```yaml
# In the experiment input
run:
  metrics: auto
  metric_interval: 50
```

`auto` uses bounded summary recording for a training policy and detailed recording
for a non-training evaluation policy (including rules). `off`, `summary`, and
`detailed` are explicit alternatives. The CLI override is
`--set experiment.run.metrics=summary`; the interval is a positive physical tick
count, not an episode or wall-clock interval. It controls cumulative-series samples,
not the recovery-analysis window.

Training rows in `episodes.jsonl` and the atomic continuation generation retain
summary metrics. Evaluation rows in `cases.jsonl` embed the detailed metrics in
the existing checksummed atomic case store. Resume reuses complete committed cases;
it does not append duplicate telemetry or silently mix source/configuration versions.
No additional checkpoint format, background recorder, service or dependency is needed.
Detailed recording is optional because per-resource and event records cost time
and space. Summary mode stores no per-tick trace.

Do not hot-patch running experiments. Historical rows can use `aggregate(row)`
only for fields already present. Missing event histories or per-resource integrals
cannot be reconstructed from a final scalar. Re-evaluate a retained model in a
separately identified, matched evaluation only when those measurements are needed;
no retraining is required to measure a frozen policy. A new evaluation is new
evidence, not a replacement for the original log. Changing runtime identity still
fails the existing source-freeze checks.

## Measurements and denominators

All times below are physical ticks unless a name ends in `_seconds`. Parent case
fields retain total orders N, delivered orders D, failed final inspections F,
observation end tick T, condition, scenario seed and original reward/cost. Outputs
use JSON null for undefined values, never infinity or a fabricated zero.

| Measurement | Recorded definition and coverage |
| --- | --- |
| Delivery and drain | D/N; `drained` only for all N delivered, excluding wall-clock fragments |
| Passing rate | Conditional `(D-F)/D`, null if D=0; final `(N-F)/N` only when drained; qualified coverage `(D-F)/N` remains valid when censored |
| Tardiness and joint cost | Full-N mean tardiness and the existing J only when drained; no completed-subset J |
| Tail delay / flow time | Nearest-rank tardiness P50/P95/P99 and mean completion-minus-arrival time, with explicit completed-order count and `distribution_scope`; incomplete distributions are descriptive only |
| On-time service | On-time/N coverage and on-time/D conditional rate, including inspection failures as delivered orders |
| Throughput | D/T and (D-F)/T, stored per 1,000 ticks; these own-horizon rates must not be passed off as common-window rates |
| Machine utilization | Processing, idle, blocked, down are mutually exclusive pre-advance state integrals; denominator is machines × observed duration; available processing excludes down time from the denominator |
| Buffers | Detailed per-machine Pre/Post resident item×tick integrals; excludes reservations; divide by recorded duration for mean occupancy |
| Vehicles | Detailed loaded/empty, moving/stationary and service-at-boundary/no-service-at-boundary duration partitions; partitions overlap each other and must not be summed together |
| Decisions | Counts by role and phase, plus STOP and empty-WAIT counts; existing `protocol`/`physics` counters remain available for retries, services and traffic diagnostics |
| Learning effort | Training rows retain episode, cumulative physical ticks, updates, losses, rollout wall seconds and PPO-update seconds |
| Scenario scale | Detailed order, operation-node, chain-precedence-edge, machine and vehicle counts, plus exact outage schedule and hashed exogenous scenario |

Vehicle service is sampled at the advance boundary. It is not a claim of exact
internal handling start/end durations. Moving time is based on endpoint displacement;
the current Base engine advances by one tick while moves/services are active.
Machine down state takes precedence; integral conservation is checked against the
engine's existing machine counters. Buffers reflect resident items at the same
pre-advance boundary. Detailed orders carry semantic IDs, arrival/due/completion
ticks, final inspection outcome and censoring status.

Detailed series contains completed/qualified/failed/unfinished counts, arrived
backlog, future-unreleased demand and observed overdue. It includes the initial
and final boundaries, every configured sample, and relevant arrival/fault/repair/
delivery events. A simulation time jump does not backdate later deliveries into
earlier samples. Fault/repair events retain machine identity, including an initial
fault at tick zero. `observer_seconds` covers before/after sampling only, excluding
decision counts, finalization and serialization; it is not total telemetry overhead.

## Fair comparisons and proposal measures

The dependency-free `smartsom.telemetry.research_comparison` API takes explicit
protocols and existing rows. It does not select a favorable target, subset, budget,
reference world or disturbance automatically. Bad identities/parameters raise
`ValueError`; absent evidence returns `not_recorded`; zero denominators or invalid
measurement applicability return `not_applicable`; unfinished observation returns
`censored`. An invalid-input exception is not a measured failure or a zero score.

1. Compare the same decoder, scenario/condition panel, training budget and complete
   training-seed roster. Report physical ticks and episodes separately: an episode
   can end at different lengths. Also report active training time on identified
   hardware/concurrency; a compute-speed result is not sample efficiency.
2. Keep sampled-policy repeats nested within their world and worlds within each
   training seed. Produce a per-seed result first; report all seeds and dispersion.
   Do not treat world repeats as independent training replicates. Do not combine
   Mac reconstructed and old server frozen source identities as interchangeable.
3. Partial panels keep their coverage, delivery, passing and censoring statistics.
   Do not silently drop failed worlds and compare different successful subsets.
   Test results never choose a checkpoint or learning target.

### SIG and learning curves

`relative_gain(base, social, direction="lower" | "higher")` returns both the
proposal's raw SIG `(social-base)/base` and a direction-aware improvement. Our J
is a cost: raw SIG < 0 is an improvement. A zero reference is not applicable.
Inputs must already be matched; this scalar helper cannot prove a causal SL gain.

`compare_cost_curves(base_points, social_points, budget_unit=...)` computes
trapezoidal cost AUC and budget-normalized mean cost on **identical increasing
budgets**, with the same training seed, decoder and validation panel identity.
Each point supplies `budget`, `cost`, `all_drained`, `train_seed`, `decoder`, and
`panel_sha256`. Missing/censored full-panel costs suppress AUC instead of removing
bad points. No extrapolation or automatic smoothing is performed. Lower AUC can
indicate better performance over the measured training budget; it does not prove
convergence. Comparing endpoints alone does not measure learning speed.

`LearningTarget(version, reference, maximum_cost, minimum_passing, consecutive,
budget_unit)` freezes an explicit joint cost/quality/delivery target. Freeze its
values and independent validation-reference identity before inspecting the target
campaign's curves; a hash establishes identity, not that preregistration occurred.
`time_to_target(points, target)` additionally requires `passing_rate` and
`model_sha256`. It returns the first qualifying interval bounded by the previous
non-qualifying observation and the first qualifying checkpoint, plus the budget
where persistence is confirmed. Repeated labels of the same model do not count as
new persistent attainment. Missing observations break persistence but do not
advance the last definite failure boundary; `evidence_gaps` preserves their presence.
Repeated-model failures also break persistence. No observed non-qualifying predecessor is represented
by a null interval lower bound. Insufficient observations remain censored.

`acceleration(base_result, social_result)` uses confirmation budgets for the same
protocol and paired seed/decoder/panel. With `episodes` this is an operational CAR;
with `physical_ticks` it is a separate sample-budget ratio. Neither is a
mathematical convergence claim. Zero-budget initial attainment has no finite ratio.
The caller must retain every seed, including censored seeds, rather than average
only the finite ratios. Native milestone validation records `training_progress`
in its frozen manifest/cases (seed, episode, ticks and active rollout+update time).
Standalone evaluation has no inferred training budget; a model's evaluation tick
count is never substituted for the training budget. Active time excludes checkpoint
I/O, validation, idle gaps and final telemetry serialization.

### Robustness and recovery

`fixed_window(row, start, end)` counts exact recorded completions in `(start,end]`
and returns throughput/qualified throughput per tick. The full window must lie
inside the recorded trajectory. Completed finite-demand runs do not invent an
unobserved zero-production tail.

`robustness(nominal, perturbed, start, end)` implements the proposal throughput
ratio on a common window. Both rows must share their exogenous scenario hash,
model, method, decoder, action-seed domain, algorithm and runtime identity
(Python, dependency versions, platform and numerical thread allocation). The
scenario fingerprint includes orders, factory, physics and random seed/sample
inputs; only outage schedule and observation horizon are excluded. Evaluation
records the actual policy seed, including learning-policy seed reduction. This
pairs the random domain; divergence in decision counts does not imply identical
sampled action sequences. Conditions alone do not establish pairing. The caller
must choose the nominal baseline and intended perturbation, and retain their exact
outage schedules, failure/coverage counts and backlogs alongside the ratio.

`RecoveryProtocol(version, reference, window_ticks, consecutive_windows,
fraction=0.95)` specifies **non-overlapping repair-anchored** windows and required
persistence, rather than treating the recording interval as a recovery rule.
`recovery(nominal, perturbed, machine_id=..., fault_tick=..., repair_tick=...,
protocol=...)` requires actual same-machine fault and repair evidence. Overlapping
faults censor the isolated-event measurement; a later fault limits observation.
Reference zero output or exhausted finite demand cannot prove recovery. Output
status `threshold_confirmed` means persistent return to the protocol threshold;
it does not claim there was a measurable performance drop first. Report the
pre-/during-fault throughput separately when making that claim. This first API
measures **post-repair** confirmation, and cannot establish immediate or pre-repair
recovery. It reports time from both fault and repair, with Ad = 1/post-repair
confirmation delay. Report this operationalization alongside the proposal formula.
Unconfirmed events retain a censor reason and observed time, never infinite/zero Ad.
This measures frozen-policy operational recovery, not online relearning/adaptation.

## Explicitly unavailable proposal quantities

Task-graph volatility requires an agreed graph-change definition; order arrivals
alone are not that quantity. Team complexity requires an interaction index and β;
raw counts are not a substitute. The recorded operation-node/edge counts can feed
an explicitly chosen task-complexity α later. Team heterogeneity also requires a
frozen definition of the agent population/types. Norm stability is inapplicable
without an organizational norm module, and energy efficiency is inapplicable
without energy physics. These are not silently reported as zero. The schema
records missing reasons for the relevant unavailable quantities; adding unsupported
physics or actor information is outside this metric change.
