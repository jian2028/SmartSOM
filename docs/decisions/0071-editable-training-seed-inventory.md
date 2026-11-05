# 0071 — Editable training seed inventory

Status: development; engineering and independent review required.

This extends ADR 0064. A frozen native directory batch may revise an authored
training seed list while all other scientific inputs and source/dependency
identities remain unchanged. The current list selects eligible seeds; removing
one does not delete results or stop a running task. Existing frozen trials and
their scientific identities remain unchanged. Newly authored seeds receive new
trials in the same experiment directory. Immutable inventory revisions preserve
their predecessor and are published in the existing queue transaction.

On a drained queue, `batch-run DIR --resume-missing` selects missing trials;
`batch-run DIR --id NAME --select 'train_seeds: [101, 102]'` selects explicit
seeds, including for an existing batch. `run FILE --select 'train_seeds: [101]'`
uses the same owning directory and selects only that Experiment. The established
`run FILE --resume-missing` selects the file's remaining authored seeds.
`run FILE --seed` remains a scientific override, not an execution selection.
An accepting running queue uses `batch-add`; existing stage-0 admission rules
continue to apply. Repeated completed tasks are not relaunched.

Changes to other scientific inputs are rejected. Runtime source/dependency
identity checks still apply; this is not automatic migration of batches frozen
under another code version. A failed preparation must be retried with the same
authored inputs before making another change. No formal experiment is authorized
by these engineering contracts.
