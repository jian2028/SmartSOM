# 0083 — Rule-calibrated finite drain deadlines for V3

Date: 2026-11-09. Status: user-authorized V3 amendment.

Supersedes ADR0082's completion-only campaign termination choice. The opt-in
all_delivered runtime capability remains available, but V3 recipes use the existing
drain_deadline mode. Actual launch exposed reachable production deadlocks; no
additional admission constraints or new failure penalties are introduced here.

For each map, evaluate Rule on 32 calibration cases: eight paired seeds times four fault
conditions in the existing calibration seed domain (71000), independent of final tests.
For each completed case, completion_tail_ticks = completion tick - last arrival.
The shared map deadline is ceil(1.5 * maximum observed completed tail). A run ends
at last arrival + drain_ticks if demands remain, and is classified as truncated.
It still terminates immediately when all demands are delivered.

Reports expose completed-tail range and explicit noncompletion/missing counts.
The recommended value is descriptive of successful calibration cases; it does
not assert completion of excluded cases. Never substitute a truncation tick for
completion. Final test data does not select the value. Calibration summaries are
recorded separately from final tests and checkpoint selection.

Explicitly write the selected value into each Small/Medium algorithm YAML, including
Base/ISL shared/independent, Central, Rule and Random. There is no shared configuration or automatic synchronization. Each recipe
authors its own limit; do not dynamically read changing Rule output during training. Re-pin test world hashes because their
horizon/outage suffix changes, preserving job inputs, seeds and action repeats.
Large retains its uncalibrated authored drain value and must not be launched.

Three reward costs, pickup rules, gamma and learner settings remain unchanged.
Incomplete makespan, final tardiness, learning_cost and completion_tail_ticks are
censored; incomplete cases remain in coverage and checkpoint eligibility logic.
Truncation uses the existing bootstrap contract rather than successful termination.
V3 display/file names are Central and Rule; historical runs remain preserved.
Changed scientific inputs require a new frozen run, not editing old continuations.
