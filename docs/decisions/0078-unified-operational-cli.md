# 0078 — Unified operational CLI

Status: implementation under engineering validation.

Supersedes the public CLI spellings in ADR0058/0065/0077. Frozen-source execution,
scientific identities, case ordering and continuation validation remain unchanged.
The user explicitly requested removal of old commands rather than compatibility
aliases. Existing running processes and archived CLIs are not upgraded or restarted.
Internal journal commands, file schemas and Python entry points are not renamed.

`run` accepts a file, directory or unambiguous name. `runs` lists runs; `status`
combines metadata and runtime observations, with optional task/receipt selection
and JSON. `pause`, `resume`, `stop` use `--task-id` for scoped queue operations.
The scientific `run --task train/evaluate/...` remains a task type, never an ID.
`retry` handles failed work; paused work uses `resume`. Active batches accept
exact-task retries, while ended batches use explicit whole-driver `retry --all`.
No failed work is implicitly retried by ordinary resume. `monitor` remains read-only
and `attach` controlling. `add` and `close` manage queue membership/admission.

`configure` receives a single validated patch and applies it under the existing
runtime transaction. It changes experiment concurrency/mode, aggregate CPU/memory
budget and/or evaluation process/CPU targets in one revision. Bad or unsupported
fields reject the complete patch. Command IDs are idempotent and expected revisions
retain optimistic concurrency checks. The current driver must advertise support.
Unspecified fields remain unchanged. Task scope permits only evaluation fields.
Task evaluation overrides inherit unspecified fields from the queue default.

Launch, check, resume and configure use `--max-concurrent`, `--concurrency-mode`,
`--evaluation-processes`, `--evaluation-cpus`, `--cpus` and `--memory-gib` consistently.
Evaluation CPUs cap the combined workers of one experiment's panel; aggregate CPUs
cap the whole selected run/queue. The new operational `concurrency.evaluation_cpus`
is excluded from scientific identity. Standalone launch honors the cap, while
live phase resizing remains a native batch capability. Logical admission quotas
are not OS isolation. Existing task resource budgets remain upper bounds.

`remove` retains task files; `clean` deletes a stopped run or removed task only
with its target-bound preview confirmation. Whole-run clean refuses active owners
and never implicitly stops them. Existing ownership/registration/deletion checks
remain authoritative. A successful stop returns 0 after verified process exit;
accepted asynchronous controls explicitly return received/applied receipts, not
a fictitious completed worker transition.

The CLI removes batch-* names, runs list/show, resume --retry-failed and stop --delete.
Option abbreviations are disabled so removed --task spellings cannot silently
resolve to --task-id. Current documentation, shell completion and tests migrate
together; historical decision records retain their original spellings.
