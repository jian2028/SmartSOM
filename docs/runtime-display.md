# Current runtime display and monitoring

`run`, `resume` and `attach` share current progress and control
records. Nested workers report into their coordinator; Studio has an independent
Qt renderer. The [current workflow](current-workflow.md) defines launch inputs.

## Display options

- `--verbose` / `--no-verbose` controls routine summaries; failures remain visible.
- `--debug` / `--no-debug` controls diagnostics independently. Numeric verbosity
  accepts 0/1 only; verbosity 2 is retired.
- `--progress auto|on|off` controls the terminal panel. Dynamic rendering requires
  an interactive terminal; text/JSON output remains available when redirected.
- `--summary-interval SECONDS` controls periodic summaries.
- `--progress-title TEXT` changes presentation, not the frozen recipe or run name.
- `--log-format text|json` controls stderr presentation. JSON has no Rich controls.

Resume display overrides are session-only. They cannot bypass source, component
or checkpoint identity checks. Scientific configuration changes require a new run.

Training episodes, committed physical ticks, current episode ticks, validation
cases and learner updates are separate counters. Sampling at 100% does not mean
that updating, saving or final evaluation has finished. Missing measurements show
as unavailable, not zero. ETA is an estimate based on observed progress and phase
costs; restarts and concurrency changes can alter it.

Current compact progress sometimes provides committed training counters without
complete execution-phase timing. Its training throughput may be shown while total
execution throughput remains unavailable. A coordinator poll is not a producer
phase boundary. Rendering frequency never creates new simulated work.

## Files and read-only monitoring

| File | Content |
| --- | --- |
| `logs/runtime.log` | Throttled human summaries and lifecycle events |
| `logs/progress.json` | Atomic current runtime snapshot |
| `logs/debug.jsonl` | Optional detailed diagnostics |
| `logs/backend.log` | Worker/framework diagnostics when produced |

```sh
smartsom monitor RUN_DIRECTORY --once
smartsom monitor RUN_DIRECTORY --log-format json
smartsom attach RUN_DIRECTORY
```

Monitor reads current snapshots without executing a policy or changing the run.
Ctrl+C detaches the monitor. Stale timestamps indicate old observations, not proof
that a process failed. Current manifests and progress ledgers retain their existing
read-only projections. Historical executable/checkpoint/recording formats use
[archived source](archive/README.md).

Attach can submit current pause/resume, concurrency and removal controls. A
received request is not yet an applied transition. Pause waits for a supported
complete checkpoint boundary; removal preserves experiment files. See the
[unified scheduler controls](unified-runtime-scheduler.md) and
[batch operations](batch-incremental-usage.md) for exact commands and ownership
checks. Live queue adoption retains its explicit admission requirements; it is
separate from ordinary batch draining.

## Validation and terminal captures

Current tests cover throttling, JSON output, missing values, nested ownership,
source-counter freshness, pause/recovery and read-only monitoring. Native learner
and replay checks are reported separately; historical SB3/RLlib reports do not
establish acceptance for current code. Local offscreen Qt tests do not establish
manual GUI or hosted-platform CI acceptance.

For a recorded Rich console, use
`smartsom.telemetry.capture.export_terminal_svg(console, title="SmartSOM")`.
The helper preserves terminal cell positions for Chinese and Latin text. Captures
are local review evidence and are not committed by default.
