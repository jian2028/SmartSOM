# 0028 — Linked preflight, calibration and terminal control

Date: 2026-09-28
Status: user-authorized implementation; engineering and research evidence remain separate.

Supersedes the v4 foreground default described in the workflow after ADR 0027.
In an interactive terminal, `smartsom run experiment.yaml` and `smartsom tune
run/recommend` launch a verified background driver and attach a controlling Rich
view. Explicit `--background` returns immediately; explicit `--no-background`
retains foreground execution. Closing or detaching the view does not terminate an
already registered driver. The separate `monitor` command remains read-only.

The phase axis shows preflight, calibration, training including periodic
validation, and final evaluation from saved evidence. It does not add unlike
units into one percentage. Its active stage has its own count or time bar;
skipped and unavailable stages are labelled. Preflight quick mode validates all
frozen inputs and policy contracts; full mode adds disposable, bounded policy
smoke execution without learner updates. Full coverage defaults to every entry.
During smoke, `p` reduces coverage from every entry to representative entries,
then to skip optional smoke. Required checks cannot be skipped. `d` detaches the
view. Two Ctrl+C presses within three seconds request cooperative task stop in
an attached controlling view. In read-only monitor they close the view only.

Calibration and full smoke are separate: smoke establishes executability,
whereas calibration measures the actual batch and chooses resources. A smoke
result is never treated as a throughput measurement. Saved `preflight.json`,
`calibration.json`, selected layout and progress snapshots provide separate
reviewable evidence. An engineering smoke or UI check does not establish model
quality or formal research results.
