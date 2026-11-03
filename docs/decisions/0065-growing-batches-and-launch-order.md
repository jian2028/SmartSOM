# 0065 — Append-only batch membership and execution order

Status: inventory/stage implementation locally reviewed; the unpublished CLI
ordering interface is superseded by ADR0066. Not yet published or deployed.

Supersedes ADR0064's fixed directory membership and its inclusion of never
submitted deferred tasks in implicit stage barriers. The original scientific
plan, manifest, child paths, source archives and completed results stay immutable.
Existing members must still have exactly the same full authored seed list,
scientific identity, dependencies, stage and task paths. Removing or modifying
one requires another output root. Runtime code/dependency checks are unchanged;
this is not a migration mechanism for runs frozen under older executable code.

New members are frozen into hash-addressed `inventory/*.json` extensions. The
queue transaction atomically publishes an ordered, parent-linked digest list in
`control/runtime.json`. Readers combine the original plan and those extensions.
An interrupted pre-publication allocation is not scheduled. A matching retry can
reuse only that preparation's verified, unstarted snapshots; it never overwrites
an unknown child or adopts an existing scientific result. Execution-state writes
by a driver started before append must retain newly accepted members and receipts.

`batch-run --resume-missing` can reopen a drained queue and append newly discovered
members while selecting only requested entries/seeds. `batch-add` supports a
single file or directory in a running/accepting queue, preserving owning-batch
paths and atomic duplicate claims. Its existing independent stage-0 restriction
remains explicit: later-stage work is submitted through batch-run after the
active driver finishes. Closed queues require batch-run, not an implicit launch.

`--id` remains selection, not an ordering contract. `--order ID` is repeatable and
must name every selected entry exactly once. It controls execution order only:
no scientific hash changes, no reordered seed identities or overwritten results.
Without it, discovery is lexical by relative path, then authored seed-list order.
Within a stage, explicit entry order precedes seed expansion; stages still run
ascending and explicit dependencies must complete. Never-submitted `deferred`
earlier tasks do not block; queued/running/failed/paused/stopped earlier tasks do.
A failed dependency needs explicit recovery and is not silently treated as done.

Appended work joins the tail of the existing admission order. Its `--order`
applies within that submission, does not reorder previously accepted work, and
never preempts a running worker. Explicit runtime/UI priority continues to take
precedence. Resource admission and task holds can defer an otherwise earlier
task; with multiple workers neither launch simultaneity nor completion order is
guaranteed. Use separate stages for a completion barrier, not an order flag.

All acceptance here uses bounded engineering fixtures, not formal simulations.
