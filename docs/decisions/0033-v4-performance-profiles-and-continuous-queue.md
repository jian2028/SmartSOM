# 0033 — V4 reusable performance candidates and continuous training admission

Date: 2026-09-29
Status: Development execution contract

This decision supersedes ADR 0032's 30-minute default and mandatory separate
group waves when no mixed measurement exists. It extends ADR 0027's explicit
uncalibrated fallback for a directory batch. It does not change Factory,
Workload, Algorithm, validation cases or formal evidence requirements.

V4 Tune and directory batches default to quick calibration with one shared
five-minute wall-clock budget. Full uses 30 minutes unless an explicit timeout
overrides it. Every expanded directory entry first receives an independent
128-tick, no-update engineering smoke. All smokes must pass before calibration;
they are outside the performance budget and cannot prove a later complete
4096-tick case will succeed. Quick probes use a disposable 512-tick case. Full
probes one representative full-length case and a formal update quantum, subject
to the same deadline; neither executes formal training or all held-out cases.

The local, locked and atomically updated `runs/.performance-profiles/index.json`
holds eligible past measurements and per-run report locations. Hardware,
allocation, learning and physical workload shape, validation/update cadence and
probe version determine compatibility. The default candidate is the latest
compatible valid profile; `best` uses historical measured throughput, and a
report path selects from that report. Selection only orders a new measurement:
old throughput never authorizes a current recommendation. Bad or incompatible
cache records are ignored, while a bad explicit report is an error. Frozen
resumes use their own report and do not query the cache again.

When quick calibration expires without a valid measurement, the original
layout is recorded as uncalibrated. Explicit probe engineering failures stop
the batch. An uncalibrated trial first runs alone to a committed update; live
resource observations then gradually open additional admissions up to, but not
beyond, the declared concurrency ceiling. No constant memory estimate may
authorize a whole batch to start at once.

All same-stage training entries on the same device use one ordered Tune queue.
Its global concurrency ceiling is the maximum declared by a stage file, not
the sum. Per-file and per-calibration-group limits and live CPU/RAM admission
still apply. When a mixed probe predicts a pair is slower together, those
groups cannot overlap. Without such evidence, conservative cross-group refill
is allowed and explicitly marked schedule-uncalibrated. Finished entries remain
finished on stop/resume; retry uses committed native checkpoints.

Dirty-source runs remain development diagnostic evidence. A formal research run
still requires an integrated main commit or explicit tag in its manifest.
