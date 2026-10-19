# 0052 — Single-node execution control and logical task removal

Status: user-authorized implementation; engineering acceptance is reported separately.

Extends ADR0024 and supersedes the opt-in/open-idle requirement of ADR0047 for
ordinary directory and Tune batches. Frozen science, native continuation identity,
and the verified workload grouping of ADR0040 remain separate from execution
control. A runtime sidecar stores operator intent, sequenced command receipts,
manual/adaptive mode, logical resource budgets and persistent removal tombstones.

The local driver and its existing shared admission journal are the scheduling
authority. Controllers submit commands through `experiments.runtime_control`;
they do not launch workers or overwrite frozen plans. One local filesystem lock
serializes command IDs, optional revision checks and state replacement. Repeated
IDs with identical payloads return the original receipt. A received or applied
intent is not proof of a checkpoint or resource release; observed worker state
and verified exit establish those facts. A future ClearML adapter must use this
same API and must not introduce a second scheduling authority or learner SDK
dependency. This decision does not implement multi-machine scheduling.

Manual concurrency is an independent-experiment target; adaptive concurrency is
an operator ceiling over existing online advice. Both retain live CPU/RAM guards.
Lowering a target drains naturally. Logical contribution budgets constrain
admission and are not operating-system isolation. Workload grouping, scientific
budgets, seeds, sampler layout and environment count remain unchanged.

Pause closes admission before requesting checkpoint-boundary worker exit. An
individual hold allows other jobs to fill the released slot. Batch continue does
not clear individual holds. A resumed task queues without preempting another.
Paused-only batches stay controllable. Batch stop saves safely and ends the
attempt; it never escalates a timeout to kill. Explicit force remains separate.
Native composable PPO/DQN continuation saves learner/optimizer, simulator,
collector, replay where relevant, clocks, validation and RNG state. This is a
complete update/validation boundary contract, not arbitrary mid-update recovery
or a cross-platform numerical reproducibility guarantee.

The primary individual terminal action is **remove from this batch**. The API
requires a target/status/revision-bound confirmation; CLI and Rich first display
the object and require a separate explicit confirmation. A running task remains
visible as removing until saved exit and reservation release. Failed saves remain
failures. Removal retains outputs, checkpoints and audit, never edits the frozen
plan, and survives restart. Removed work cannot be continued under its old task
identity. Summary outcomes distinguish partial removal from all tasks succeeding.

Ordinary batches accept additions while active and finish when drained; explicit
keep-open retains an idle service. End-of-queue and append acceptance are serialized
so accepted work is not lost during shutdown. Added science is frozen separately
and expanded into independent native entries. Historical explicit adoption queues
retain their compatibility protocol; ordinary dynamic control needs no adoption.

CLI monitor is read-only. Attach offers explicit controls, stable focus and
pagination, a compact summary and expandable details. Confirmation captures an
immutable selection; sorting, refresh, empty input or repeated hotkeys cannot
retarget or confirm removal. Non-TTY removal requires the explicit batch, task,
confirmation token and revision rather than a generic yes switch.
