# 0024 — Explicit CLI tasks and local cooperative stopping

Date: 2026-09-27
Status: User-authorized implementation; verification is reported separately.

Extends ADR 0013 and 0020–0023 with a unified driver entry and control protocol.
Scientific configuration, strict/adaptive continuation identities, simulator
ownership and historical entry semantics remain unchanged.

`check` and explicit `run --task` resolve the same selected task and input kind.
Training fields do not infer a task. Existing v3 single, native prepared Study
and Tune batch executors are reused. Compatibility entries remain available.
Check is read-only and cannot establish live execution capacity or successful
scientific outcomes.

Execution control records are separate from frozen inputs. Cooperative stop
halts admission, reaches native training/update or evaluation tick boundaries,
preserves available recovery/partial evidence, and releases execution resources.
It is distinct from completion, early stopping and failure. Native single-run
resume retains its training-only boundary; this decision does not add a workflow
stage ledger or relax source checks.

Explicit force may terminate only verified owned process identities after a
bounded cooperative wait. PID alone, process names and stale registration never
authorize a signal. A shared batch driver is stopped at its top-level directory.
No control command may terminate an unrelated application, worktree experiment
or shared Ray cluster. Old unregistered active processes are not retrofitted.

Foreground ownership remains the default. Monitoring is read-only and does not
maintain a process or detach it from a terminal. Local POSIX control and CPU
engineering qualification do not establish Windows, cluster or GPU acceptance.
