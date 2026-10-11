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
- `--progress-title TEXT` overrides the panel title for the current session.
  It does not rename the run or change its frozen recipe.
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

New V4 online `run` and `batch-run` sessions show two workflow stages: preflight
and experiment execution. Manual/off runs also omit the isolated performance
stage. Explicit quick/full calibration and recommendation sessions retain it.
The execution panel contains an online performance card: running experiments,
current admission limit/hard ceiling, confirmed concurrency and the latest
complete throughput window with its measured concurrency. Throughput is total
committed physical ticks divided by a shared wall clock, including scheduled
validation and save overhead; it never advances the scientific progress bars.
The latest complete window is labelled as a past observation, rather than an
instantaneous speed at a newly increased limit. Cached hints and retained history
do not become fresh measurements after restart. Waiting for online resources
keeps the execution view and experiment cards, without probe counters or budgets.
Short terminals use a compact performance card; all group observations remain in
the text log and snapshot.

The configured workflow panel refreshes at most once per second when its state
changes. Legacy panels refresh at most four times per second. Sampling,
learner updates and saves update the existing panel rather than appending a
new table each iteration. Major phase transitions and final outcomes receive
immediate text/file summaries. Sampling at 100% is not a completed run: the
run remains active through final writes and any requested evaluation.

Training, resume, evaluation and train-evaluate use a shared workflow layout.
Its top bar covers the declared training budget, scheduled validation horizons
and requested evaluation horizons. An ended case resolves its planned horizon;
the active case contributes its elapsed fraction. This planned-work estimate
combines adapter decisions and case horizons for presentation; it is not a sum
of physical simulation ticks or a claim that each unit has equal execution cost.
Training-only omits final evaluation, and evaluation-only omits training cycles.
Older snapshots and unconfigured run/batch/search sessions keep their existing
counter display. The renderer also accepts explicit study metadata from a
coordinator; this change does not add a study execution command to main.

Validation and evaluation have separate current-case bars, using counters from
the actual driver about every two seconds and at case boundaries. Validation
does not advance the training budget. Collection/update boundaries and learner
updates remain separate; optimizer minibatch counts appear only when reported.
Unknown counters remain `N/A`. Updating and saving show elapsed phase time
when there is no reliable percentage. Failures remain separate from completion.

Elapsed time and ETA refer to this display session. ETA shows `estimating` until
at least 30 seconds of advancing work and then uses up to five minutes of recent
wall-clock progress. Phase costs, concurrency and early case completion can
change this approximation. Cached progress does not imply fresh throughput.

Interactive training/evaluation sessions use the alternate terminal screen,
overwriting the panel in place. The original scrollback returns on exit with a
final text summary. Wide terminals show instance cards; compact terminals use
a table and current-phase bar. Hidden active tasks are explicitly counted.

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
Each task also records its start time for its own approximate ETA; older
snapshots without this field continue to show `估算中`.
Optional `workflow` metadata describes the frozen budgets and presentation
titles; `overview` records planned-work progress, elapsed time and approximate
ETA. Monitor validates and consumes these fields using the same renderer.
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

## Configurable titles

```yaml
logging:
  title: "My map comparison"
  task_title: "{algorithm} · Seed {seed}"
```

`title` sets the panel title and `task_title` sets instance titles. Templates
support `id`, `name`, `algorithm`, `H`, `V`, `travel` and `seed`; missing values
are `N/A`. H/V are optional coordinator labels. Templates are single-line text;
unknown fields, attribute access, conversions and format specifications are
rejected. Titles remain archived presentation metadata and do not change the
main grid recipe's scientific identity. Existing source and checkpoint checks
still apply.

```sh
smartsom train --config configs/test/runs/learning_sb3.yaml --progress-title "PPO training"
smartsom evaluate RUN_DIRECTORY --progress-title "Held-out evaluation"
```

Python callers can use `display_options={"title": "My title", "task_title":
"{name} · Seed {seed}"}`. A title override never bypasses source validation.
# Unified lifecycle controls

Use `smartsom stop RUN_DIRECTORY` from another terminal to request a safe stop.
The default timeout reports remaining processes; `--force` is explicit.
`monitor` also shows stopping/stopped control state and remains read-only.
See [command workflow](command-workflow.md) for process ownership and resume limits.

If the runtime display's output stream fails during a write or flush (for example,
a disconnected pipe), that session stops mirroring output to that stream and
continues writing progress and logs. `logs/progress.json` records `console_error`
on the next snapshot or session close. Task failures and evidence-file errors
still propagate. The display borrows its stream and does not close it or replace
process-wide stdout/stderr; the caller remains responsible for those streams.
This does not guarantee process survival after external termination or terminal
closure, nor change the CLI's separate stdout result contract.
