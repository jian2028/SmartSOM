# Single-node runtime scheduler

Runtime control manages the number of independent experiments. It does not
change PPO sampling environments, scientific budgets, seeds, rewards, validation
plans or frozen experiment configuration. New ordinary directory batches and
Tune batches expose one shared control interface. The existing `--live-queue`
adoption path remains a compatibility interface; it is not required for new
batches. This is a single-node implementation, without a ClearML service or SDK
dependency in training algorithms.

See the [cloud engineering validation record](runtime-scheduler-validation.md)
for executed checks and the remaining platform/CI boundaries.

## Start and inspect

```sh
uv run smartsom batch-run EXPERIMENT_DIRECTORY --max-concurrent 4 --background
uv run smartsom batch-status RUN_DIRECTORY
uv run smartsom monitor RUN_DIRECTORY --once --log-format json
uv run smartsom attach RUN_DIRECTORY
```

Use the returned run directory for subsequent commands. A normal queue accepts
additions while alive and exits when its runnable work drains. Pass `--keep-open`
to `batch-run` when it should wait for later additions. A paused task remains
unfinished: a batch containing only paused tasks stays available for continuation.
`batch-close` closes admission for additions and clears keep-open, then lets
accepted work finish. It does not resume paused tasks.

```sh
uv run smartsom batch-add RUN_DIRECTORY ADDITIONAL_EXPERIMENT_DIRECTORY
uv run smartsom batch-close RUN_DIRECTORY
```

Additions are frozen when accepted and must be independent stage-0 work. Task
IDs must be unique. An addition does not rewrite the original scientific plan.
Use `batch-status` to inspect actual task IDs, including expanded entry IDs.
`batch-add --command-id REQUEST_ID` makes retries return the original receipt;
`--expected-revision` rejects stale submissions before allocating more work.
Default inferred file ordering is not a dependency. Explicit staged additions,
dependencies and evaluation gates must be submitted as a separate frozen batch.

## Concurrency and resource protection

```sh
uv run smartsom batch-limit RUN_DIRECTORY 6
uv run smartsom batch-limit RUN_DIRECTORY 2
uv run smartsom batch-mode RUN_DIRECTORY manual
uv run smartsom batch-mode RUN_DIRECTORY adaptive
```

Manual mode treats the operator limit as a target. Adaptive mode treats it as an
upper bound on measured admission decisions. The displayed mode identifies what
`batch-limit`, `+`, `-`, and numeric entry change. Group performance estimates are
separate from operator intent. A target does not promise that enough tasks or
resources exist to fill every slot.

Tune retains its measured online throughput feedback. Native appended workers
use resource-headroom adaptation: their effective admission count is bounded by
CPU capacity and observed peak memory, without changing the operator ceiling or
claiming a measured throughput improvement. Both share the same slot and logical
budget journal, including reservations still awaiting process exit.

Native singleton, matrix and Tune requests share a persisted stage resource model in `control/resource-model.json`. Actual worker samples distinguish startup, sampling, optimizing, validation, evaluation and saving. The semantic resource group retains resolved workload, policy/model and runtime shape, while excluding documented seed, display-name and redundant input-path metadata. The implementation digest invalidates old evidence after code changes; this grouping does not change checkpoint/scientific identity.

The raw memory request is the maximum of 256 MiB, declared memory and sampled stage RSS peaks. When any expected stage lacks a CPU or memory observation, it is multiplied by 1.5. Shared admission applies its existing 1.25 RAM factor. CPU requests retain the declared runtime shape and at least 1.1 times sampled stage CPU. Same-phase CPU windows accumulate until 0.1 seconds; shorter observations remain unknown. These sampled estimates cannot guarantee capture of every transient allocation.

All shared slots contribute their future envelope, even when the operator has not set a budget. Free host capacity can only add back recently observed owned usage with matching live process identity; missing/stale usage adds nothing. Reserved future memory is never treated as current usage. Model-backed reservation highwaters survive new observations and coordinator restoration. The temporary whole-host reservation for an uncalibrated first Tune worker is bounded by both the local broker and shared envelope. It is recorded separately from the persistent resource floor and releases only after a verified committed update with measured memory. Repeated synchronization retains the real floor without restoring the temporary excess.

Resource reservations are logical admission protection, not operating-system isolation. Existing running learners retain their allocations until verified exit. Native matrix entries share the runtime limit and are admitted between observations. Tune throughput grouping and operator concurrency remain separate from the resource envelope.

A ready request that can fit the shared node envelope receives waiting priority
after 30 seconds. New smaller work then waits for sufficient capacity to drain;
existing workers are not preempted. Feasibility includes verified current usage
from both native and Tune owners. Paused/removed or presently impossible work does
not block the queue. This avoids same-node backfill starvation under the tested
conditions; it is not a fixed completion deadline under external resource pressure.

Set the node contribution budget independently of concurrency:

```sh
uv run smartsom batch-budget RUN_DIRECTORY --cpus 4 --memory-gib 8
```

At least one positive finite value is required. Memory is converted from GiB to
bytes; omitted fields retain their existing setting. The budget is durable and
applies to manual and adaptive admission. Its compact display explicitly labels
it as a logical cap.

Both modes retain CPU and memory admission protection. Lowering concurrency lets
existing workers finish naturally; `draining` identifies an actual count above
the target. It does not kill or restart those workers. Resource headroom,
dependencies and safe checkpoint transitions can delay admission. Current
blocking reasons and actual counts are available in `batch-status` and monitor
details. CPU and memory contribution budgets are logical scheduling limits,
**not operating-system isolation**. They cannot enforce a hard RSS cap on a
running learner.

Online performance grouping retains the verified learner shape contract in
[ADR0051](decisions/0051-online-learner-shapes-and-stable-monitor.md). Comparable
seeds can share estimates; frozen partners and different workload/runtime/model
contracts remain distinct. The operator ceiling is not Tune's search concurrency
or a request to modify PPO sampler geometry.

## Pause, continue and stop

| Command | Meaning |
| --- | --- |
| `batch-pause RUN` | Close replacement admission, then request safe checkpoint pauses |
| `batch-resume RUN` | Continue the batch; preserve individually paused/stopped tasks |
| `batch-pause RUN --task ID` | Pause one task; other queued work may fill its released slot |
| `batch-resume RUN --task ID` | Queue that task for continuation without preempting another worker |
| `batch-stop RUN` | Cooperatively stop the batch and exit its scheduler |
| `batch-stop RUN --task ID` | Compatibility action: stop one task; retain its files and explicit stop intent |

`pausing` means workers have not all reached their safe boundaries. `paused`
means execution is held. `stopping` is not a successful completion. Stopped tasks
do not wake on a global continue. An explicit task resume is required; if the
batch driver has already exited, use its existing run-type resume command first.
The saved runtime settings and task holds survive driver recovery. A batch with
stopped tasks must not report every task as successfully completed.

A slot is released after the worker has saved and exited, not merely after the
request is received. This mechanism does not use `SIGSTOP` as a claim of freeing
memory. Native PPO/DQN continuation uses existing complete training checkpoints,
including weights, optimizer, counters, random states and algorithm-specific
rollout/environment or replay state at supported boundaries. Pause waits for a
supported committed update/episode boundary; **arbitrary-instruction restoration
is not supported**. Scheduled validation and evaluation follow their existing
safe recovery contracts. Unsupported backends cannot claim complete restoration
from weights alone.

Exact checkpoint deserialization and equal remaining scientific budgets are
separate from cross-platform numerical reproducibility. Changing hardware,
framework versions or floating-point execution can change subsequent numerical
results even when the checkpoint state is faithfully restored. Source identity
and checkpoint compatibility validation remain in force.

### Waiting for a safe transition

Rich keeps pause, resume and removal feedback in a fixed two-line area. Durable
receipt, applied intent and observed completion are distinct. While a pause is
pending it shows a spinner and elapsed waiting time; remaining time is unknown.
It says “正在安全暂停” when individual checkpoint/exit stages cannot be reliably
observed. The summary counts pausing separately from paused. Stale/unavailable
observations use the fixed panel border so the task cards do not shift. A long wait or stale scheduler observation is shown as such, without
inventing a deadline, declaring success or killing a worker.

A global operation freezes its target IDs when the scheduler applies it. Later
additions do not change that denominator. Paused, naturally completed, failed and
pending targets remain distinguishable; one failed target does not hide other
pending transitions. Details retain each target's last observed status. Repeated
pause requests while pending reuse the operation; repeated Space during that
wait does not resume it. Held/buffered Space events are debounced. After confirmed
completion, a separate Space action can resume the batch. `a` cycles recorded
operation feedback without submitting commands, including in read-only monitor.

These indicators report observations, not a bound on pause duration. Safe pause
can wait for the current update/episode, scheduled validation, checkpoint saving
and process exit. Measure request receipt separately from final paused state;
small development fixtures do not predict latency for large experiments.

## Remove a task with explicit confirmation

Removal is a scheduling tombstone, not deletion of checkpoints, logs or scientific
artifacts. A running task remains visible as `removing` until safe cleanup and
process exit; only a confirmed `removed` task leaves the active cards. The summary
retains its removed count. Removal cannot be triggered with a generic `--yes`.

Interactive CLI:

```sh
uv run smartsom batch-remove RUN_DIRECTORY --task EXACT_TASK_ID
```

The command displays the batch, task snapshot and impact, then requires typing
the exact displayed short ID. Blank input, mismatch, EOF or interruption cancels
without submitting a removal command. The revision and target state are checked
again on submission; a changed target requires a fresh review.

For scripts, first inspect the bound target without changing it:

```sh
uv run smartsom batch-remove RUN_DIRECTORY --task EXACT_TASK_ID --show-target
```

Then explicitly provide the returned values:

```sh
uv run smartsom batch-remove RUN_DIRECTORY --task EXACT_TASK_ID \
  --batch-id RETURNED_BATCH_ID --expected-revision RETURNED_REVISION \
  --confirm-target RETURNED_CONFIRMATION_TOKEN --command-id UNIQUE_REQUEST_ID
```

The token binds the task, revision and state. A stale token is rejected. Do not
substitute a token from another task or batch. Permanent file deletion is outside
these commands.

## Rich and noninteractive monitoring

`monitor` is read-only; `attach` enables control. The fixed summary covers every
submitted task, including those off-page: successful completion/total, running,
queued, paused, failed and removed counts, actual/target concurrency, mode and
draining. Batch CPU/RSS and host-available resources are separate observations.
Training counters do not imply workflow completion; incomplete ETA stays unknown.

Each experiment has six fixed information rows: algorithm/seed/status,
update/budget/elapsed time, phase progress, observed speed/CPU/RSS, latest observed
argmax/sampling return with update and validation round, and a reserved warning.
Speed belongs to the current phase: validation/evaluation use their own tick rates.
Training cumulative counters do not establish sampling or optimizer progress.
Optimizing/saving without a recorded denominator remain unknown.
Missing observations remain unknown. Current resource observations older than five
seconds are marked stale; sampled peaks are not current usage. Return is a
reported metric, not a replacement for the configured model-selection objective.

Tasks retain submission order. Width determines columns and height determines
pages; selection and page anchors survive resize. Enter opens a separate scrollable
detail view containing recorded learner diagnostics, phase timing/work/rates,
sampled resource peaks, saved checkpoint path/update and validation aggregates.
A saved checkpoint path is not a claim that restore validation has succeeded.
Unavailable data is explicitly unrecorded. No Textual dependency is introduced.

| Key | Action |
| --- | --- |
| Up/Down or `j`/`k` | Select a task; scroll in detail |
| Left/Right or `[`/`]` | Previous/next task page |
| Enter or `i` | Open selected task detail |
| `a` | Cycle recorded operation feedback; observation only |
| Esc | Return from detail; cancel current input/confirmation |
| `+` / `-` | In attach, increment/decrement global target/adaptive ceiling |
| `c`, digits, Enter | In attach, submit an absolute concurrency value |
| Space | In attach, pause/continue the entire batch |
| `P` / `R` | In attach, pause/resume the selected stable task ID, including from detail |
| `D` | In attach, review removal of the selected task |
| `d` | Detach without stopping work |
| Ctrl+C twice within three seconds | Stop the batch in attach; close read-only monitor |

Mode switching (`batch-mode`), CPU/RAM budgets (`batch-budget`), task addition
(`batch-add`) and queue closure (`batch-close`) remain CLI operations. The legacy
per-task stop command remains available in CLI only. Pause/resume/stop accept
`--task ID`; omission acts batch-wide. The existing `p` key adjusts preflight
smoke coverage, not scheduling mode. `batch-status` retains complete task state
and command receipts; navigation never submits scheduler commands.

`P` and `R` act on the selected task ID, not its page position. A pause waits for
a safe checkpoint and verified worker exit before queued work can fill its slot.
Resume queues that task under the current global pause, concurrency and resource
budgets; it does not preempt another worker. Global continue preserves manual
per-task holds.

Uppercase `D` freezes the selected target for review. Type its displayed short ID
and press Enter to submit; Escape, EOF, blank or wrong text cancels. Repeated
`D`/`d` keystrokes are not confirmation. While confirmation is open, keys belong
to its input, and changing selection cannot redirect the pending action.

EOF detaches without stopping the scheduler. Non-TTY output has no Rich terminal
control sequences. Interactive changes are unavailable without terminal input;
use explicit CLI commands and JSON snapshots instead. Transient unreadable
snapshots retain the last valid display; stale observations remain labelled.

## Adapter and audit contract

All new frontends use `smartsom.experiments.runtime_control`:

- `snapshot(root)` reads state without creating files.
- `submit(root, kind, ...)` receives an identified command under a filesystem lock.
- `removal_target(root, task_id)` returns a bound confirmation snapshot.
- `append(root, directory, ...)` freezes additional independent work.

`control/runtime.json` contains operator settings, revision, task holds, command
receipts and scheduler observations, separate from frozen inputs. Commands are
serialized and idempotent by `command_id`; reusing an ID for different content
is rejected. `--expected-revision` supports compare-and-submit. Rich increments
submit deltas, avoiding read-modify-write loss between multiple windows.

`received` confirms durable receipt. `applied` confirms application of control
intent, **not** that a worker has already paused or a requested slot has started.
Use actual task state, active count and blocking reason to distinguish progress
from resource waiting. `rejected` carries its reason and CLI returns unsuccessful
status. Future ClearML adapters must use the same API and must not become a
second scheduler authority.

## Reproducible engineering checks

Run from the repository root with the locked dependencies appropriate to each
suite. These are test commands, not a claim that a formal experiment or every
acceptance scenario has passed:

```sh
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run pytest \
  tests/unit/test_runtime_control.py tests/unit/test_live_queue.py \
  tests/unit/test_runtime_display.py tests/unit/test_batch_dashboard.py \
  tests/unit/test_tuning_dashboard.py tests/unit/test_workflow_dashboard.py -q
UV_CACHE_DIR=/tmp/smartsom-uv-cache uv run pytest \
  tests/integration/test_tuning_native_continuation.py \
  -k scheduler_pause_across_processes -q
```

The frontend checks cover read-only behavior, terminal EOF, atomic delta routing,
receipts, exact-target removal confirmation, tombstone visibility and fixed Rich
geometry. Actual learner/process acceptance, available cloud resources and any
remaining unsupported cases must be reported with their executed test results.
Do not infer six concurrent PPO learners from a lightweight six-process test.

## Historical layout comparison (before the card redesign)

The following text captures document the earlier compact layout. Current card
acceptance renders cover 1/4/12 tasks at 60×24, 80×24 and 160×50; executed results
and source identity belong in the engineering validation record. These historical
captures are not images of the current interface.

The following before/after views are rendered from deterministic display-test
fixtures at 100 columns by 24 rows. They illustrate layout only; their task and
resource values are not measurements from a training run.

### Previous detailed default

```text
┌──────────────────────────────────────── SmartSOM · tune ─────────────────────────────────────────┐
│ ╭────────────────────────────────────────── 整体流程 ──────────────────────────────────────────╮ │
│ │ ○ 预检           未开始                                                                  N/A │ │
│ │ ✓ 性能评估       已完成   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% │ │
│ │ ▶ 训练＋最终评估 进行中   ━━━━━━━━━━━━                                                 21.0% │ │
│ ╰──────────────────────────────────────────────────────────────────────────────────────────────╯ │
│ ╭───────────────────────────────── 当前阶段 · 训练与最终评估 ──────────────────────────────────╮ │
│ │ 正式实验结束 0/4 · 成功 0                                                                    │ │
│ │ 训练 1680/4000 ticks · 实验完成 0/4                                                          │ │
│ │ 实际资源：等待运行实例上报                                                                   │ │
│ ╰──────────────────────────────────────────────────────────────────────────────────────────────╯ │
│ 批次预计剩余 采集中；后续阶段观测不足                                                            │
│ ┌─────────────────────────────────────── trial-0 · 采样 ───────────────────────────────────────┐ │
│ │ 训练预算 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                              42.0% │ │
│ │ 当前采样 ━╸               10% 本轮20/200 physical ticks                                      │ │
│ │ 420/1,000 physical ticks · ETA 采集中；训练： 速率采集中 · 待生效                            │ │
│ │ CPU 实际/请求 2/8 · 资源请求将在安全边界应用                                                 │ │
│ │                                                                                              │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────────┘ │
│ 实验卡片 · 第 1/4 页 · 任务 1–1/4 · 1 列                                                         │
│ [ 上一页 · ] 下一页 · d 离开界面 · Ctrl+C×2 关闭监控                                             │
│                                                                                                  │
└─────────────────────────────────────── 正式实验 [running] ───────────────────────────────────────┘
```

### Unified compact default

```text
┌──────────────────────────────────────── SmartSOM · tune ─────────────────────────────────────────┐
│ 成功 0 · 失败 0 · 暂停 0 · 已移除 0 · 任务 4                                                     │
│ 控制 · 手动目标 4 · 实际运行 4 · running                                                         │
│ 可控制 · +/- 调整 · 手动目标 · CPU/内存预算保护（非操作系统硬隔离）                              │
│ CPU N/A/N/A 核 · 可用内存 8 GiB · 吞吐 N/A ticks/s                                               │
│ 逻辑预算 · CPU 8 核 · 内存 8 GiB（非 OS 硬隔离）                                                 │
│ i 查看阻塞原因、命令结果和控制记录                                                               │
│                                                                                                  │
│                                                                                                  │
│ ╭──────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ ▶ trial-0 · running                                                                          │ │
│ │ 420/1,000 physical ticks · ETA 采集中；训练： 速率采集中                                     │ │
│ │ sampling                                                                                     │ │
│ ╰──────────────────────────────────────────────────────────────────────────────────────────────╯ │
│ ╭──────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │   trial-1 · running                                                                          │ │
│ │ 420/1,000 physical ticks · ETA 采集中；训练： 速率采集中                                     │ │
│ │ sampling                                                                                     │ │
│ ╰──────────────────────────────────────────────────────────────────────────────────────────────╯ │
│ 实验卡片 · 第 1/2 页 · 任务 1–2/4 · 1 列                                                         │
│ [ 上一页 · ] 下一页 · +/- 并发 c 数值 空格 暂停 D 移除 j/k 选择 i 详情 d 离开 Ctrl+C×2 停止批次  │
│                                                                                                  │
└──────────────────────────────────────────── running ─────────────────────────────────────────────┘
```
