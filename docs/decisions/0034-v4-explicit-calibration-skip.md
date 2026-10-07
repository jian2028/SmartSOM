# 0034 — Explicit V4 calibration skip

Date: 2026-09-29
Status: Development execution contract

This extends ADR 0033 for a user-selected fixed-layout run. `execution.calibration_level:
off` is an explicit third value alongside `quick` and `full`. It skips performance
probes and history selection, not V4 input checking, per-entry engineering smoke,
rule gates or Tune training. The frozen runtime layout and declared concurrency
ceiling become an uncalibrated execution report with zero measurements and a
zero calibration budget. Resume uses that report. A nonzero timeout combined with
`off` is invalid.

An uncalibrated group admits one formal trial first; after a committed update
and observed process memory peak, the existing resource broker can increase
admission toward the frozen ceiling while checking live CPU/RAM. The profile is
not described as calibrated or optimal. The Rich workflow bar shows calibration
as pending before a scheduled probe and explicitly skipped for `off`; neither
state appears as completed during preflight. The formal batch bar uses a short
label that fits narrow terminals.
