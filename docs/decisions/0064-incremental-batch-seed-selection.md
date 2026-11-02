# 0064 — Incremental batch execution selection

Status: user-authorized implementation; independent review and CI pending.

Extends ADR0061's authoring/output layout. Canonical variant/experiment.yaml
entries remain supported. A batch directory may also contain named Experiment
YAML files; schema detection excludes Scenario and Algorithm files.

`--first-train-seeds N` is execution selection over the authored training seed
list, not a scientific override. The complete directory inventory and full seed
lists are frozen before execution; unselected tasks are deferred. `--id` selects
entries, never implicitly starts other entries. Scientific `--set train_seeds=1`
retains its original meaning: a seed count expanding from 101.

`--resume-missing` verifies the bound absolute authoring directory, complete
inventory, referenced files/components, frozen child plans and runtime identity
before activating missing selected tasks in the same output root. It can reopen
a drained queue without an earlier `--keep-open`. Completed trials remain intact;
training with unfinished final evaluation uses the native continuation/evaluation
runtime. Failed tasks require explicit retry; started training without complete
continuation is rejected rather than restarted. Manual task holds survive.

The immutable plan describes all tasks; an execution-state selection describes
this invocation. A finished selection with deferred work reports `partial`, with
separate completed/deferred/failed counts. Stage barriers include earlier stages;
a later selected stage cannot bypass unfinished earlier-stage tasks. Reports
rebuild from distinct trial and evaluation identities, never append observations.

A persistent submission lock spans validation, selection, and driver ownership.
Foreground execution holds it through completion; background execution holds it
through the existing verified process-owner handshake. Later submissions check
live ownership before touching state. No distributed scheduler or cross-host
locking is introduced.

`batch-add` accepts one Experiment file, a directory or a resolvable name. The
owning batch directory determines the result destination, independently of the
submitted file/directory selection. Named external batches default to a path-bound
subdirectory of the project's runs tree, never the source input directory. A
batch's explicit output root remains bound to that original directory.

The native queue may schedule another batch's stage-0 tasks at their existing
trial paths. The original immutable batch manifest remains authoritative. Existing
queue-admission ownership claims prevent the original batch from launching a
second driver; native completion updates its execution state and releases the
drained claim. Repeated file/directory submissions deduplicate task identities.
A crashed queue retains its claim until explicit queue recovery; no stale-PID
heuristic transfers ownership. Closed queues still require explicit batch resume,
not an implicit background launch from `batch-add`.

Engineering fixtures do not authorize formal runs.
