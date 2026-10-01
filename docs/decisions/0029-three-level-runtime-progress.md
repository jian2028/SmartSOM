# 0029 — Three top-level runtime progress bars

Date: 2026-09-28
Status: user-authorized implementation; engineering display evidence only.

Supersedes the four-stage phase-axis presentation in ADR 0028. Keep three
full-width bars at the top of the Rich view: preflight, performance calibration,
and training plus final evaluation. The active bar is highlighted. Completed
bars remain visible; skipped and unavailable phases are labelled. The overview
has its own frame, followed by a current-stage detail frame. Calibration shows
its measurement and resource facts there, without displaying future training
cards. Once formal execution starts, one experiment card per visible trial shows
sampling, updates, periodic validation, and final evaluation. Normal terminals
stack cards; wide terminals use two columns. Short terminals condense cards and
state how many trials are hidden. The same hierarchy applies when calibration
is skipped.

The bars have distinct evidence and denominators. Preflight counts required
checks and optional smoke dispositions, with skips displayed separately.
During calibration, the bar shows consumed wall-clock budget, not the fraction
of candidate configurations measured. If calibration converges early, its bar
fills as a **completed stage** while showing the actual elapsed time and number
of measured candidates. The formal bar gives each experiment equal weight and
each applicable training/final-evaluation phase equal weight within that
experiment. Each phase is first normalized against its own physical-tick or
evaluation-case budget; unlike units are never added. Planned experiments not
yet started contribute zero. A recommendation-only run marks the formal bar
unexecuted. Compact terminal views retain the three bars and prioritize the
active case and hidden-experiment notice over secondary resource detail.

These percentages are UI progress indicators, not throughput or model-quality
measurements. Saved preflight, calibration and per-experiment progress remain
the authoritative counts.
