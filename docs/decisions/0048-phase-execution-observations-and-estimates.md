# ADR0048: Phase execution observations and remaining-time estimates

Status: accepted for implementation by the PR22 user request.

Runtime progress carries an optional `smartsom.execution-observation/v1` DTO.
Its producer attempt and sequence identify observations across restore, actor
replacement and nested projections. It stores accumulated phase-active seconds,
measured work and recent phase rates. Monotonic absolute timestamps remain in
memory and fresh processes establish new counter baselines. Scientific plans,
checkpoint parameters and scheduler decisions are not changed by estimates.

Training, validation, final evaluation, initial/shared controls, loading,
checkpointing and finalization retain separate observations. Explicit waiting
and pause intervals are excluded from active phase clocks. Real validation
trigger/completion/pending events determine outstanding validation obligations.
A training budget is not proof of evaluation or artifact completion. Missing
required rates produce a partial estimate with a reason, never a zero-duration
phase. Display refresh and pagination do not create measurements.

A small heap model estimates current submitted tasks under occupied slots,
concurrency drain, dependencies and stage barriers. It preserves long tails and
uses published CPU/RAM reservation facts conservatively. Unknown pause/resource
wait durations produce conditional service estimates. Open queues cover only
currently submitted work. There are no inferred confidence intervals and ETA
need not decrease monotonically as better observations arrive.

Native, Tune and directory/queue projections share the saved observation DTO.
The aggregate execution throughput includes measured training, validation,
evaluation and control tick deltas; training throughput contains only training
ticks. Both divide by the same coordinator's recent monotonic wall window.
Attempt baselines exclude historical restored work and repeated nested producers
are counted once. Warming, zero, missing, corrupt and stale observations remain
distinct. Old snapshots without phase timing safely warm up; training-only
counters cannot establish execution-total throughput.

Restored durable completion facts are separate from freshly measured counters.
Unknown counter baselines stay unknown until a real counter or verified restore
state arrives. Completed training/validation does not recur in a final-only resume
estimate. Full workflow plans merge with partial presentation overlays, so control
names cannot erase horizons or matching cost groups. Initial controls publish
their own phase; native final writes and shutdown establish finalization costs.
Directory and native parent drivers sample their own current CPU and execution
window, and directory stage changes discard preceding-stage throughput.

CPU presentation distinguishes visible host busy cores from the batch-owned
process tree. Owned identities require matching births, and optional identity
sampling failures remain unavailable rather than zero. Monitor and attach read
the same contract without writing telemetry or control files.
Visible CPU busy/total share the host/affinity range; effective cgroup/SLURM
capacity is separately labelled. These new display fields do not alter resource
admission. Compact pages reserve control notices; monitor with redirected input
uses complete output without inaccessible navigation.

The implementation follows the baseline and rolling-window practices documented
by [tqdm](https://github.com/tqdm/tqdm/blob/master/tqdm/std.py) and
[Rich](https://rich.readthedocs.io/en/stable/_modules/rich/progress.html), while
keeping phase accounting in the application. Actual concurrency can be lower
than a declared cap, as described by
[Ray Tune's resource guide](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html).
The runtime adapter remains pinned to Ray 2.58; these references do not relax that
compatibility contract.
