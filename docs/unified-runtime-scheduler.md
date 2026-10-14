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

`monitor` is read-only; `attach` enables control. The default unified view contains
stable outcome counters, concurrency/mode, compact resources, lean task cards and
shortcuts. Detailed blocking reasons and recent command results are under `i`.
The detail rows reserve their height, so opening details does not move task cards.
`batch-status` retains the full command history and task state.

| Key | Action |
| --- | --- |
| `+` / `-` | Atomically increment/decrement target or adaptive ceiling |
| `c`, digits, Enter | Submit an absolute target; Escape cancels |
| Space | Global pause/continue in attach |
| `j` / `k` | Select the next/previous task |
| `[` / `]` | Previous/next task page |
| `i` | Toggle diagnostics; allowed in read-only monitor |
| `D` | Open removal confirmation for the selected task in attach |
| `d` | Detach from the view |
| Ctrl+C twice | Stop the batch in attach; close read-only monitor |

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

## Rendered layout comparison

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
