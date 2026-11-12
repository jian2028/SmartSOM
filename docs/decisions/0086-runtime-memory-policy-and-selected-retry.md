# 0086 — Configurable native memory policy and selected driver retry

Status: authorized implementation; engineering validation recorded separately.

Extends ADR0077 and supersedes ADR0078's ended-batch `retry --all` restriction.
Scientific inputs, source matching, checkpoint hashes and physical semantics remain
unchanged. Existing processes and archived sources are not automatically upgraded.

Native resources expose `memory_headroom_ratio` (default 0.25, range [0,1]) and
`host_memory_reserve_ratio` (default 0.20, range [0,1)). Experiment YAML, operation
profiles, run/check/resume/retry flags and atomic configure use the same names.
Both are queue-wide operational settings, not per-task scientific parameters.
New native drivers advertise the capability; old drivers reject live changes.
Nondefault ratios reject Tune compilation rather than silently changing only part
of its resource policy. Explicit stopped-driver overrides stage validated resource
commands for restart; ordinary live configure still rejects an ended driver.

Admission retains the absolute 2 GiB host reserve and existing resource guards.
Memory charges retain unscaled high-water evidence and reapply the current factor;
lowering the extra margin does not discard observed usage. Evaluation shrinkage
still waits for live workers before releasing CPU leases. Within-panel historical
peaks remain conservative; this change does not introduce inference-only loading
or infer that old peaks are safe to discard. Panel closure resets the baseline
only after evaluation children have joined.

`retry --task-id ID` is repeatable and works for native failed tasks after their
driver ends. Validate all targets, frozen selection, dependencies, ownership exit,
complete recovery hashes and RNG state before mutation and again under the driver
lock. Only selected failed tasks are queued; other failures, paused/held tasks,
deferred seeds and their artifacts remain unchanged. Selection is execution-only;
the authored inventory and scientific plan are not rewritten. Overrides supplied
to selected retry update the queue and selected children, not unselected children.

A recovery scope is recorded in runtime state and the background request. Failed
peers remain visible, so the whole batch can still report failed after the selected
retry completes; the return value separately reports retry_status/retry_tasks.
Pending retries outside that scope must be settled first. New live retry requests
outside a selective recovery are rejected until that driver ends. Ordinary resume
clears the recovery scope. `--all` keeps its existing whole-driver behavior.

External-owner/nested retry and historical source migration are not expanded.
Hard RSS guard failure is distinct from memory admission: adjust guard settings
through a validated retry profile when appropriate; lowering headroom alone does
not change a max_worker_rss_gib limit.
