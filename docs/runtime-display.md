# Runtime display and logs

Mainline `run`, `train`, `evaluate`, `train-evaluate`, `resume`, `batch`,
`batch-train` and `search` share one Rich display per top-level call. Nested
training/evaluation and worker processes do not create competing panels.
Studio's graphical renderer remains independent.

For batches and searches, `Finished tasks` counts tasks that reached a final
state, including failures, ineligible candidates, pruning and interruptions.
The summary reports successful tasks separately. Pending or queued tasks remain
unstarted when a session is interrupted; they do not count as finished. Individual
sampling bars describe sampling budgets, not successful validation or saving.
The dashboard shows active tasks, a queued count, and up to three recent ended
results, fitting the terminal height. Older results remain in `logs/runtime.log`;
the latest failure remains identified even when it leaves the recent list. Ended
tasks have result counters and reasons, not partially filled progress bars. An
evaluation stopped by its decision limit reports that reason and the decision
count independently of physical ticks. Training status and evaluation successes
are reported separately. Reaching a sampling target does not finish a task that
is still updating or saving. Names, states and unit-bearing counters occupy
separate lines so narrow terminals do not squeeze them into competing columns.

## Terminal controls

- `--verbose` enables human-readable summaries. `--no-verbose` hides routine
  summaries and the panel; failures remain visible.
- `--debug` enables detailed diagnostics independently of verbose. `--no-debug`
  explicitly disables them. Numeric verbosity 0/1 remains accepted; legacy 2
  enables debug and emits one deprecation notice, unless debug is explicitly
  overridden. New configurations should use booleans and `logging.debug`.
- `--progress auto|on|off` controls the dynamic panel. Both `auto` and `on`
  require an interactive terminal. `off` retains periodic text summaries.
- `--summary-interval SECONDS` overrides `logging.every_seconds`; the new
  default is 30 seconds. Existing explicitly configured intervals remain valid.
- `--log-format text|json` controls stderr presentation. JSON mode emits one
  versioned runtime snapshot per line, plus typed debug/warning records when
  requested. It never emits Rich controls. stdout keeps the existing command
  result contract (including the text result of `run`).

Display options on a resumed run are session-only overrides. They do not rewrite
its frozen configuration or bypass checkpoint/source validation. Changing the
implementation still changes the source identity: this feature does not allow
old frozen experiments to resume against different code.

The Python API supports `display_options={"verbose": True, "debug": False,
"progress": "auto", "every_seconds": 30.0, "format": "text"}` as a keyword on
runtime entry points. Nested entry points borrow their parent's display. Ordinary
recipe logging settings remain supported. Display does not change callback
return values used by pruning or stopping.

## What the panel means

The panel refreshes at most four times per second during execution. Sampling,
learner updates and saves update the existing panel rather than appending a
new table each iteration. Major phase transitions and final outcomes receive
immediate text/file summaries. Sampling at 100% is not a completed run: the
run remains active through final writes and any requested evaluation.

Counters retain their units: environment/adapter decisions, physical ticks,
agent steps, physical actions and actual PPO updates are distinct. Evaluation
progress counts finished attempts; successful episodes are a separate counter. Training cumulative deliveries and the latest finished
episode are separate from evaluation outcomes. A missing value is `N/A`, not
zero. In particular, historical checkpoints may lack the information required
to reconstruct cumulative training deliveries; that field remains unavailable
on such a resume. Metric names and signs such as `entropy_loss` are retained.
Roles are obtained from actual metric scopes, never filled from a fixed list.

Multiple tasks have separate rows and progress bars. Total completed tasks is
separate from each task's sampling budget. Narrow terminals use fewer columns;
when tasks exceed terminal height the panel prioritizes active tasks and gives
an omitted-task count. The final text log retains all task summaries.

## Files and monitor

Runtime presentation adds these files under the top-level run/study directory:

| File | Purpose |
| --- | --- |
| `logs/runtime.log` | Timestamped, throttled human summaries suitable for `tail -f` |
| `logs/progress.json` | Latest atomic `smartsom.runtime-progress/v1` snapshot |
| `logs/debug.jsonl` | Detailed display diagnostics when debug is enabled |
| `logs/backend.log` | Framework stdout/stderr diagnostics (also shown with debug) |

The progress snapshot contains run identity, overall phase/status, task IDs,
per-task units/targets/counters, selected learner metrics and update times.
It is presentation data, not a checkpoint or scientific evidence ledger.
Snapshots are written at most once per second except lifecycle transitions.
`logs/events.jsonl`, episode ledgers, full learner metrics, checkpoints and
configured TensorBoard/W&B streams retain their numeric evidence independently
of summary throttling. Workers keep their own `worker.log`; coordinator output
never forwards these raw logs wholesale. Optuna informational chatter is quiet
unless debug is requested. tqdm is neither replaced globally nor uninstalled.

```sh
smartsom monitor RUN_DIRECTORY
smartsom monitor RUN_DIRECTORY --once
smartsom monitor RUN_DIRECTORY --log-format json
smartsom monitor RUN_DIRECTORY --summary-interval 60
```

Monitor reads once per second, uses the same Rich renderer, and exits after a
recorded final state. `--once` reads and prints one snapshot. Ctrl+C exits only
the monitor. Noninteractive output uses the summary interval and contains no
terminal control sequences. A stale timestamp means the data is old, not that
the process has failed. A transient unreadable/partial replacement retains the
last valid snapshot. Initial invalid input produces an error.

Older mainline directories without snapshots expose only fields actually
available in their manifests and the bounded tail of their training event log.
Monitor does not modify them, acquire their locks, load a learner, or add data.
W38 base-learning/base-readiness, dispatch pilot and overnight formats are not
supported by this feature. Their launchers, running processes, frozen sources,
archives and experiment data are outside this change.

## Coverage and verification

| Entry | Display owner / input |
| --- | --- |
| run | Run session; committed simulator state |
| train / resume | Training session; original evidence event stream and read-only episode projection |
| evaluate | Evaluation session; individual runs and recorded evaluation results |
| train-evaluate | One outer session across both phases |
| batch | Parent coordinator; worker progress queue and final attempt states |
| batch-train / search | Parent coordinator; existing trial queue and final trial results |
| monitor | Read-only snapshot consumer; no execution/control path |

Tests cover throttling without evidence loss, JSON/redirection, dynamic roles,
missing values, failure/interruption/pruning, nested ownership, quiet workers,
read-only monitoring and legacy options. Real small SB3, RLlib and resource PPO
runs compare display-on/off trajectories, weights, optimizer state and RNGs.
Framework timing metrics remain recorded but are excluded from equality checks.
These checks establish engineering behavior, not experiment performance or
milestone acceptance. Terminal visual verification is recorded separately from
unit assertions.
