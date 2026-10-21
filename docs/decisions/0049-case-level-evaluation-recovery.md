# ADR0049: Case-level evaluation recovery

Status: user-authorized implementation; engineering verification is separate.

Supersedes the whole-stage evaluation retry portion of ADR0025 and the
training-only single-run resume boundary of ADR0024 for new V3/V4 evaluation
records. Training continuation, simulator physics, rewards, scientific budgets,
source checks, and historical result semantics remain unchanged.

An evaluation freezes an ordered case plan before executing its first case.
Case identity contains the complete materialized world and recorded seed streams,
policy/composition bindings, hashes of actual inference payloads, effective
reward and evaluation protocol, implementation identity, and stage. The total
count is an obligation of the frozen plan, not part of an individual case ID.
Changing count, order, model, input or seeds while recovering that plan fails
closed and requires a new evaluation. Recovery never generates a new seed from a
compacted list of unfinished cases or counts files as observations.

Each completed execution writes its recorder and result, then atomically publishes
one checksummed case receipt with fsynced contents and a POSIX directory fsync.
Windows cannot establish directory fsync; file contents are still synced. A
receipt is the sole case completion authority. Only after its publication may
progress advance. Valid truncated cases are observations and are retained;
engineering failures and interrupted cases are not reusable completion. Missing
receipts cause only their original cases to rerun, preserving unfinished evidence
in a separate interrupted-attempt directory. Corrupt or mismatched receipts are
errors, never successful observations. Unpublished temporary files are ignored.
The final summary is rebuilt in frozen order from each verified identity once.
A crash after the last receipt but before summary publication needs no simulation.

Local advisory locks from the existing experiment locking module reject concurrent
writers for one stage. The lease covers evaluation and summary writes; a separate
case-store lease also protects direct native case evaluation. This adds no
scheduler, database, cross-host recovery or inside-case simulator checkpoint.

`evaluation.replications` is the final total per generated scenario and may be
zero. `initial_control` retains its explicit enable switch.
`initial_control_replications`, when supplied, is an independent nonnegative
initial-control target; absent it retains the historical final target. New
preparation freezes enough ordered worlds for both stages. Each stage selects its
own unchanged prefix. Explicit frozen-input suites keep their authored final
case inventory; an optional initial target selects a verified prefix of that
inventory. A larger initial target than that inventory is refused.
Initial and final models have distinct stage identities, directories, receipts,
results and counts. INITIAL-CONTROL and last-diagnostic ranking exclusions are
committed with the case result, including its evidence reference, before reuse.
Resuming a standalone evaluation starts only that saved stage.
The empty-input precondition uses that stage's own target. A zero-target final
can resume and rebuild its summary even when the independent initial target is
positive; it never starts the initial stage as a side effect.

V4 native plans keep stable final/initial evaluation directories beneath each
entry's runs folder, allowing recovery even if the parent stage ledger update
was interrupted. Training stages are still skipped or resumed under their prior
checkpoint contract. Tune uses the same native receipts in an experiment-owned hidden recovery
directory shared across fresh `attempt-*` allocations. Restored inference
payloads must match before reuse. Completed attempts receive ordinary relative
evidence copies for reports and portable adaptive commits; hidden recovery
evidence is excluded from report discovery. Case commits newer than a training
checkpoint are preserved rather than rolled back with training support files. Existing complete Tune/control
artifacts retain their established whole-stage verification path.
Shared initial-control cache selection and plan discovery occur under the
shared control lock. A waiting attempt must verify any plan published while it
waited, including when it already has a legacy local metrics copy.

Runtime snapshots publish historical committed, current committed, remaining and
total counts, together with per-stage progress. Scientific successful episodes
remain distinct from committed observations. ETA consumes the exact remaining
horizons and current active-case ticks with ADR0048 phase rates. Restored case
counts cannot create tick deltas or be divided by a new attempt's elapsed time.
Monitor and attach consume saved facts without writes.

Legacy complete results remain readable by reports, replay and audit. Old partial
case result files have no strong receipts and cannot be silently promoted into
this protocol. Strict source/dependency guards continue to apply to old saved
runs. Recovery from a new implementation is not a migration of an old experiment
or permission to start a formal scientific run.
