# 0069 — Cooperative native worker resource protection

Status: implemented in a proposed patch. Integrated review, CI and the final
1280-tick post-start performance gate remain separate acceptance evidence.

ADR0058 separates operational settings from scientific identity. This adds an
opt-in protection policy to that operational contract; admission budgets retain
their existing meaning. It does not alter the historical stage-two 80 GiB
admission threshold or introduce a custom experiment driver.

`resources.guard` defaults to null. When supplied, it requires explicit positive
finite `max_worker_rss_gib`, `min_host_available_gib`, and `min_disk_free_gib`.
The narrow-layout recipe uses 8, 8 and 2 GiB respectively, preserving the old
rule-DQN worker guard. `resources.memory_gib` continues to mean admission budget,
not an RSS limit. Guards are excluded from scientific identity with the other
resource settings and retained in resolved execution metadata and frozen inputs.

The native train/evaluate/calibrate scope owns one worker guard, including nested
final or controlled-recovery evaluation. The existing control boundary checks it
at most once per second during rollout. Model construction/loading, update
boundaries, checkpoint serialization/publication and case receipt commits force
fresh observations before and after their operations. Nested evaluation reuses
the worker's guard and cannot weaken its limits. No learning or simulator module
imports experiment orchestration. Tune and smoke reject this opt-in configuration
until they have equivalent coverage; defaults are unchanged.

Execution updates validate the complete merged Experiment for every frozen and
appended child before writing any execution history or runtime command. An
invalid child rejects the whole requested update. Loading an execution sidecar
also revalidates the effective configuration; the saved execution entrypoint
checks compatibility before binding an owner, returning a completed result or
dispatching Tune. An explicit correction may remove an invalid guard left by an
older writer, without changing frozen science or previous results.

The sample is current-process RSS, host available memory and free space on the
output filesystem. RSS reaching its limit, available memory or disk falling
below its floor, or unavailable observations raise `ResourceGuardExceeded`.
Failure evidence records the reason, byte thresholds, actual readings, phase,
PID and time. The exception is latched within that worker scope and bypasses
automatic task retries even when a general retry budget is nonzero. A fresh,
explicit recovery attempt rechecks limits and restores existing committed work.

A failed checkpoint publication leaves the preceding atomic continuation pointer
and its generation intact. A failure after pointer publication retains the new
valid generation. Completed evaluation receipts remain authoritative; unfinished
cases have no successful receipt. Resource interruption is an engineering
failure, not a truncated scientific outcome or completed panel. No emergency
checkpoint is attempted under resource pressure. Failure-report persistence is
best effort if the output filesystem itself cannot be written.

This is cooperative protection, not OS isolation or a hard peak-memory guarantee.
One long tensor operation, policy update or serialization may cross a threshold
before its next boundary check. There is no asynchronous interruption inside
atomic publication. The old guard's cooperative limitation remains explicit.
Polling costs one clock check at an existing rollout boundary and a small local
resource sample at most once per second, plus forced phase checks. No performance
percentage is claimed until the integrated 1280-tick comparison is measured with
the guard enabled. The current user-requested 5% limit excludes one-time startup
but includes all subsequent work and final save/close; startup and total time
are still reported separately.

Validation uses mocked resource readings and tiny native transactions, without
large allocations. It covers threshold inclusivity, periodic throttling, disabled
behavior, scientific identity, no automatic retry, checkpoint recovery, final
evaluation and ordinary/controlled-recovery receipt preservation.
