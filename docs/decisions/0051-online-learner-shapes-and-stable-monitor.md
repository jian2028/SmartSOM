# ADR0051: Comparable learner seeds and stable monitor geometry

Date: 2026-10-14
Status: local implementation; runtime verification remains separate.

This refines the performance-group classification in ADR0040. Scientific
identities, model payload verification, training continuation, physical rules,
reward, budgets and resource protection remain under their existing contracts.

Online performance grouping may compare independent trainable model packages
with different initial weights and training/validation seeds when their verified
effective model contracts, provider, architecture, observation implementation,
physical contract, workload shape, reward, optimizer, runtime layout and
validation/checkpoint schedules agree. Weight bytes, encoder state and seed/run
provenance are not compute shapes. Only groups explicitly selected for training
receive this normalization. Frozen model partners retain their fingerprints;
historical, incomplete or mismatched effective model contracts retain strict
keys. Quick/full probe keys retain their prior behavior. The advisory online
shape has an explicit v2 discriminator; existing scientific inputs are never
rewritten or merged.

Comparable seeds share one online controller. They still start at one real
learner, require observed process RSS and two stable committed-update windows,
grow one at a time and back off at the existing gain/pressure boundaries.
Different reward, architecture, frozen partner, sampler or validation-mode
contracts remain separate. A group with three tasks cannot establish four-way
throughput; a configured global ceiling does not manufacture another task or
prove optimal concurrency.

Entry resource reasons belong to that entry. Successful admission clears its
obsolete wait; another entry's latest wait is never projected onto a running
worker. Group-level online decisions remain available in the performance card.

Rich reserves diagnostic space and recovery space within a stable execution
stage. CPU/throughput samples, pending changes and disappearing wait notes do
not move trial card anchors or change page bounds. Short terminals keep their
primary counters and allocation visible; detailed explanations remain in the
structured snapshot and logs. Genuine stage or terminal-size changes may reflow.

New behavior applies only to new runs using this implementation. A running
source worktree remains frozen; follow-up changes live in a separate real Git
worktree until that source is no longer active. This does not add hot concurrency
control for frozen Tune batches. Live queue limits remain a separate contract.
