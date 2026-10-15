# Runtime display and logs

Mainline `run`, `train`, `evaluate`, `train-evaluate`, `resume`, `batch`,
`batch-train` and `search` share one Rich display per top-level call. Nested
training/evaluation and worker processes do not create competing panels.
Studio's graphical renderer remains independent.

Within an execution stage, resource diagnostics and recovery details reserve
their display rows. A CPU first sample, a throughput note or a disappearing wait
reason does not move task cards or reshuffle pagination. Narrow views keep
progress and CPU allocation ahead of optional explanations; full structured
facts remain available through `monitor --once --log-format json` and logs.
An admitted worker no longer displays its prior resource wait or another
entry's last wait. Terminal resizing and actual stage changes may reflow.

For batches and searches, `Finished tasks` counts tasks that reached a final
state, including failures, ineligible candidates, pruning and interruptions.
The summary reports successful tasks separately. Pending or queued tasks remain
unstarted when a session is interrupted; they do not count as finished. Individual
sampling bars describe sampling budgets, not successful validation or saving.
Tune and directory-batch dashboards retain all task cards in stable pages. Other
legacy dashboards show active tasks, a queued count and up to three recent ended
results within the terminal height. All results remain in `logs/runtime.log`;
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
when Tune or directory tasks exceed terminal height, the panel offers task pages.
Other legacy panels prioritize active tasks and show an omitted-task count. The
final text log retains all task summaries.

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

### Unified runtime controls

New ordinary `batch-run` and Tune batches use the shared runtime control API;
`--live-queue` is not required. `batch-limit RUN N` changes independent experiment
concurrency, `batch-add` freezes additions, and `batch-pause`/`batch-resume` hold or
continue a batch or an explicit `--task`. Normal queues exit when drained;
`batch-run --keep-open` waits for future additions. Paused work stays unfinished.

The default view keeps a three-line whole-batch summary above a compact table:
status counts, a weighted workflow progress bar with ETA, and concurrency/resources.
ETA uses short units (`20 min`, `1.5 h`) without an approximation prefix;
sub-minute work shows `<1 min`. Header counts, CPU/RAM and concurrency reserve
fixed value widths. The ordinal column reserves digits for the whole batch, so
moving to a page containing three-digit task numbers does not shift other columns.
Each task occupies one row: ordinal, status, task name, current phase, total progress
and ETA. Alternating backgrounds distinguish tasks before selection; focus adds a
full-row highlight and a quiet `›`. At 80 columns algorithm and seed have separate
columns and each task gets an eight-cell progress bar. At 60 columns a static
quarter-circle gauge and exact percentage replace the bar. At 120 columns phase
and phase percentage separate, with CPU/RAM; at 160 columns the current phase is
highlighted among phase labels, progress bars grow, and elapsed time/rate appear.
Wider terminals also show warnings. Fixed column widths prevent percentage changes
from moving adjacent columns. Both 60×24 and 80×24 windows fit 12 tasks; `f` switches
to all tasks, including successful tasks. Shorter windows paginate.
The bottom has a selected-task/feedback context, one reason/feedback line, the
filter and page range, and highlighted browsing shortcuts. Empty feedback stays
blank; details remain available with Enter. This region has a fixed height so
search, filtering and receipts do not move tasks or alter page capacity.

In `attach`, `Q` opens the queued-task view. Up/Down select; Enter adds/removes the
selected task in the manual next-task list; Left/Right move it earlier/later in
that list. The list is displayed above the shortcuts. Esc returns to the previous
filter and selection. Prioritized tasks precede automatic ordering when eligible;
resource, dependency and pause guards still apply. Running tasks are not preempted.
Started tasks leave the list. Requests bind an exact task ID and observed revision;
a stale selection requires reselection. Read-only `monitor` never submits changes.

Task total progress weights the frozen training, validation and evaluation/control
physical-tick budgets; batch progress sums the same work across submitted tasks.
It is work completion, not a time percentage. Unknown budgets remain unknown;
reaching the tick budget is capped at 99% until verified task completion, including
save/finalization. Failures retain work already completed without implying success.

ETA prices remaining work per phase, plus measured loading/checkpoint/finalization
costs. A separate exponential rate with a 30-second time constant weights recent
active work; at least 10 active seconds are needed. The measured rate remains
unchanged. Pauses reset the ETA window; compatible `eta_group` history can price
unobserved future phases. Queue simulation accounts for active tasks, concurrency,
dependencies, barriers and manual priorities. A queued task's ETA includes its
wait until completion. Missing rates remain `估算中`; paused tasks show `待恢复`;
`*` marks an estimate conditional on resumption/resource availability. Open queues
cover only currently submitted tasks. These are engineering estimates, not a
validated prediction-accuracy claim.

Design references: [tqdm's exponential rate smoothing](https://github.com/tqdm/tqdm/blob/v4.67.1/tqdm/std.py),
[Rich's recent-window remaining-time calculation](https://github.com/Textualize/rich/blob/v14.1.0/rich/progress.py),
and [aria2's rolling speed window](https://github.com/aria2/aria2/blob/release-1.37.0/src/SpeedCalc.cc).
No dependency or source implementation from those projects is added.

Batch-owned CPU/RSS is separate from host-available resources. Training completion
and workflow completion remain separate. Per-mode validation return displays the
actual observed update; missing results stay unrecorded. Current task resource
observations expire after five seconds, while sampled peaks remain historical.

Arrow keys or `j`/`k` select tasks; Left/Right or `[`/`]` change pages. Enter or `i`
opens a separate scrollable detail view, and Esc returns to the same selection.
The task ID and status stay pinned while scrolling; failures show their complete
reason first. PgUp/PgDn and g/G move through details, and r toggles the raw
snapshot. Normal details show only relevant learner sections and readable
resource observation ages. Press ? for full keyboard help; the ordinary footer
keeps browsing actions visible. / searches ID/name within the current view,
e toggles failure-only filtering, and n jumps to the next failure (clearing
search). Enter applies a search and an empty query clears it. Input and help
consume their own keys without submitting task controls (help retains the
two-press Ctrl+C stop/exit confirmation). Blank/zero concurrency
input stays editable with a validation message; Esc or Ctrl+C cancels input.
Removal confirmation uses a separate panel with the bound target and impact;
arrow keys scroll long confirmations without changing the typed short ID.
Current scheduling reasons remain visible beside operation feedback; a completed
operation reports elapsed time without an unknown remaining-time label.
In attach, `+`/`-` submit global concurrency deltas, `c` accepts an absolute number,
and Space pauses/continues the entire batch. Uppercase `P` pauses the selected
task and `R` queues its continuation, including from detail. These act on its
stable ID; global continue preserves individual holds. A paused worker must save
and exit before its slot can be reused. Resume obeys concurrency/resource budgets.
Lowercase `d` detaches. Uppercase `D`
locks the selected removal target and requires typing its exact short ID. Escape,
EOF or mismatched input cancels. Removed tasks retain their files and remain in
summary counts, but leave the visible cards only after verified removal.
Mode/budget changes, additions and queue closure use CLI commands. The legacy
per-task stop command remains CLI-only; Rich exposes pause, resume and removal.

Read-only `monitor` never submits controls. EOF detaches. Noninteractive mode
uses text/JSON, with explicit CLI controls for changes. `received` and `applied`
refer to the command and control intent; actual worker state and resource wait
reasons determine whether a transition has completed. Resource budgets are
logical admission protections, not OS memory isolation. Pauses wait for supported
complete checkpoint boundaries, not arbitrary instructions.

See [commands, confirmation rules, recovery limits and rendered layout examples](unified-runtime-scheduler.md).

### Live manual queues

This section describes the retained explicit adoption interface. Its older
request ledger and keep-open behavior are separate from the default new-batch
control path above.


A currently running, independent one-entry-per-file V4 training batch can be
included in a manual queue without stopping its existing learners:

```sh
smartsom batch-run ORIGINAL_EXPERIMENT_DIR --live-queue --adopt-run RUN_DIR --max-concurrent 3 --background
smartsom attach QUEUE_DIR
```

The returned `QUEUE_DIR` becomes the control and aggregate display directory.
The original run keeps its checkpoints and source identity. The transition keeps
its original driver until the adopted work finishes; later additions use the
queue coordinator and independent native workers, not additional batch drivers.

An active source must have been started with the complete shared-admission v2 protocol and
use `off` or `online` calibration. Active older drivers and `quick`/`full`
calibration are rejected because their probe/scheduler admission cannot safely
join the shared limit. A fully completed older batch whose verified processes
have exited can still be included. One active source can belong to only one
queue; an owner generation change is rejected. Frozen inputs and checkpoints
stay intact.

Smoke, pending Tune actors, running learners, retiring sampling children and new
native jobs reserve slots through the same atomic control sidecar. Queued files
without an allocation do not count as running. If adoption or a lower limit finds
more existing reservations than requested, they drain without preemption and
no additional work starts until there is room.

```sh
smartsom batch-add QUEUE_DIR NEW_EXPERIMENT_DIR --id new-task-101 --id new-task-102
smartsom batch-limit QUEUE_DIR 5
smartsom batch-close QUEUE_DIR
```

Only independent stage-0, one-entry-per-file additions are supported initially.
Inputs are frozen at addition time; changing their YAML afterward does not change
the admitted work. Task file stems must be unique. CLI commands return request
IDs; their application is recorded in the queue's `batch.json` request ledger.

In `attach`, press `c`, type the total concurrency and press Enter; Escape
cancels input. Lower limits wait for existing work to finish. `monitor` remains
read-only. The display distinguishes measured CPU busy/total logical cores from
worker thread settings, and reports rolling total execution ticks/s (training,
validation and evaluation) alongside training ticks/s over the last 30 seconds.
Observed counters exclude adopted history and include wall-clock saving waits;
polling can omit case-ending ticks when a new case starts between samples. An idle queue remains open
for future additions. `batch-close` closes admission and drains accepted work.
`stop QUEUE_DIR`, foreground Ctrl+C and exception cleanup propagate cooperative
stop to the original owner even during smoke or while all its files are queued.
The queue reports its terminal state only after that owner and all new workers
have exited. It sends no force signals. A drained queue releases admission
ownership, allowing the original batch's ordinary resume command to continue
from its own committed checkpoints. Public live-queue resume remains unsupported.
While the queue is still open, original-batch resume is rejected before changing
its control owner, including sources that finished before adoption. Cohort and stage exits
drain their verified actor/child leases before later smoke or training begins;
full Tune preflight smoke uses the same slots as directory smoke.

Background startup checks platform support before adoption. A failure that
creates no process withdraws only that failed queue's claim, preserving the
original driver's live leases so adoption can be retried. A registered coordinator,
an already created process or uncertain startup keeps the claim; the launcher
does not infer safe rollback from a missing acknowledgement.
When an owner has recorded that its scope finished, ordinary stop waits for its
registered child identities to exit without signalling a departed owner. An
unknown registration or an incomplete ownership record still fails closed.


### Task card pages

Interactive Tune, live queue and directory rule-evaluation views show task cards
in a Rich grid. Narrow terminals use one column; wider terminals add two, three or
more columns only when each card has at least 48 character cells available. Actual
rendered row height determines page capacity. Unified runtime tasks instead use
one table row per task at every width, retaining identity, status and the selected
marker. Narrow windows hide optional columns; full observations remain in task
details and noninteractive output retains full metric cards.
The footer shows the current page, visible task range and total task count; legacy
card grids also show column count. Press `[` for the previous page and `]` for the next page in either read-only
`monitor` or controlling `attach`. Existing `p`, `c`, `d` and two-step Ctrl+C controls
keep their behavior; pagination is suspended while editing a concurrency limit.

Tasks keep their first-seen order, including completed and queued tasks. Page
navigation changes only local display state, and the current task anchor survives
new arrivals, completion and terminal resizing. Pagination never writes an admission
request or changes a checkpoint. `--once`, redirected output and JSON retain all
task records, with no interactive page truncation. This remains the existing Rich
keyboard loop and adds no Textual dependency.
Control prompts and errors retain a reserved notice row even in compact pages.
The same height budget applies before any task cards exist, including calibration
and recommendation screens. Transient interaction results expire after five
seconds; a stale-snapshot warning stays visible alongside an active prompt.
When monitor has no usable terminal input, it publishes complete text instead of
opening a paged Live view whose later pages cannot be reached.

### Remaining time and measured execution

Task cards and the batch header estimate remaining phase service time from fresh
producer observations. Restored ticks become the new attempt's baseline. Training
and validation/evaluation clocks are separate, and explicit pause/resource waits
do not reduce measured training speed. Validation uses the real trigger and
completed/pending ledger. Required final evaluation, initial controls and saves
remain outstanding after the training budget reaches 100%.
Durable completed-stage ledgers seed remaining-work facts separately from measured
work. Metadata alone cannot establish a zero training counter. Tune workers retain
their full admitted workflow, including validation/evaluation horizons and the
matching cost group, before adding control names. Native reports, writer shutdown
and terminal stage-ledger writes emit actual finalization boundaries.
Native queued entries publish their complete phase/control plan from frozen
prepared inputs before worker startup, so comparable completed entries can price
their outstanding work without treating evaluation-only jobs as training.

Missing phases show `采集中` or an explained partial estimate. Paused/resource-wait
completion is conditional, and an open queue estimates only submitted tasks.
Batch estimates respect occupied slots, queue order, stage barriers and dependencies
rather than dividing all task durations by a nominal parallelism value. Estimates
are coarse, may rise after new evidence or added tasks, and never affect admission,
pruning, training parameters or checkpoint results.

The execution card labels host-visible busy CPU separately from owned batch CPU.
The busy-core denominator is the same host/affinity-visible range used to sample
its numerator. Effective cgroup/SLURM capacity remains a separate field; admission
continues using the original effective allocation. Directory and native parent
drivers sample current execution facts, and a new directory stage starts a fresh
throughput window instead of reusing its preceding Tune stage's performance.
This applies after fixed/manual as well as online Tune stages. Producer identity,
rather than the retained scheduling-policy flag, determines who samples metrics.
Live queues explicitly own their sampler even though they use the directory
rendering profile: resource ownership covers queue and adopted-source roots,
and the generic directory sampler cannot overwrite or sample those facts again.
Native stages include CPU and throughput in complete text and redirected-input
fallbacks as well as Rich and JSON output.
Total ticks/s includes observed training, validation, evaluation and controls;
training ticks/s uses only training. Both use the same recent monotonic window.
Cold or failed/stale samples explain their state; an actual observed zero stays
zero. Monitor/attach consume the saved observation contract. Historical snapshots
with only training counters cannot establish execution-total throughput.
See [ADR0048](decisions/0048-phase-execution-observations-and-estimates.md).

New case-resumable evaluations display historical committed cases, this attempt's
committed cases, remaining cases and the frozen total. The fields
`evaluation_historical_completed`, `evaluation_current_completed`,
`evaluation_remaining`, `evaluation_target` and per-stage `evaluation_case_progress`
are preserved in JSON and forwarded through directory batches. Final evaluation
and initial control keep separate counters. Scientific successful episodes remain
separate: a valid truncated case is a completed observation. ETA uses only missing
case horizons and fresh phase rates; history never creates new throughput.
`monitor --once` remains read-only. See
[case recovery](four-file-workflow.md#case-level-evaluation-recovery).

### Task visibility in the runtime monitor

Interactive runtime tables default to **unfinished/failed**: running, queued,
paused and transitional tasks remain visible, as do failures. Successful and
other terminal tasks remain in global counts and recorded results. Press `f`
to cycle **unfinished/failed → all → history**; `e` toggles **failures only**;
history includes all terminal
outcomes, including failures and removals. This is available in both read-only
`monitor` and controlling `attach`; filtering never submits scheduler commands.

Submission order stays stable within each view; filtering happens before height
pagination. Use brackets/left/right to page and up/down or j/k to select. A
selected task stays selected while visible; if it leaves the view, the next
visible task in submission order (or the last remaining task) becomes the visual
selection. Controls do not silently follow that automatic change: explicitly
select or switch views before controlling another task. A removal confirmation
keeps its original task and revision even if status changes.

An open detail view stays bound to its task ID after completion; Escape returns
to the filtered list. Empty views show an all/history hint. Failures retain a
red card border and global failure count. Noninteractive/once output retains
all task records. Existing final-batch auto-exit is unchanged; history browsing
is available while attached to a live owner (including keep-open batches).
