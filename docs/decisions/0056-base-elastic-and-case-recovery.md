# 0056 — Base elastic execution and case-level stage recovery

Status: user-authorized implementation; verification evidence is separate.

Supersedes ADR0055's fixed-profile and unfinished calibration/evaluation restrictions.
Scientific parameters, Mover, samples, seeds, updates and deadlines remain frozen.

Base now uses the existing AdaptiveBroker, PresetSearchAlgorithm and
SafeResourceChangingScheduler through build_tuner. There is no second admission
controller. Measured CPU/thread alternatives and live CPU/RAM pressure govern
resize and pause/resume at complete rollout/update boundaries. The broker's existing
hysteresis and restart-amortization guard apply. Base forecasts remaining physical
ticks against measured per-learner tick rates; other algorithms retain update units.
The authored max_concurrent remains a ceiling, not permission to exceed live guards.
Worker PID/birth/allocation-epoch acknowledgements establish applied resources.
Resume uses a fresh execution segment and authoritative native generations, as the
project's preset scheduler does; historical Ray experiment state is not replayed.

Evaluation and drain calibration reuse CaseStore receipts and stage_lock. Stable
case descriptions, input/model digests and seeds identify results. Completed and
scientifically truncated cases are both committed and never retried to improve
outcomes. An interrupted uncommitted case restarts from its seed; committed cases
are skipped. The original manifest wall deadline includes downtime/replay. This is
case-boundary recovery, not an arbitrary simulator snapshot.

Performance calibration reuses ProbeSupervisor with durable, checksummed measurement
receipts. A resumed controller reuses completed measurements explicitly marked as
historical, then measures missing work within the original performance deadline.
No observed throughput is manufactured, nor is a timeout promoted into a result.

References: Ray ResourceChangingScheduler changes allocation after a training step
and requires checkpoint-capable trials; TorchElastic restarts workers on membership
changes and relies on application checkpoints. We use these recovery boundaries,
not TorchElastic distributed-gradient training: Base seeds are independent trials.

- https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ResourceChangingScheduler.html
- https://docs.pytorch.org/docs/stable/elastic/agent.html
