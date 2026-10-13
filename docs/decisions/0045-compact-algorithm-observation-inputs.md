# 0045 — Compact algorithm-specific observation inputs

Date: 2026-10-13
Status: Authorized implementation; CI and integration pending

Supersedes ADR0041's equal-width context planes for the opt-in pair
`builtin.public_fields@2` / `builtin.scoped_candidate@2`. Version 1 retains its
encoding and network shapes. Historical default and physical-job selectors
retain their implementation digest. Source-pinned public-fields packages still
reject a changed implementation digest; this is not cross-source continuation.

DQN stores only the selected actor/Q context. PPO stores an ordered compact
actor context followed by its compact critic context; shared fields occur in
both scopes. Component values, normalization, aggregation, role applicability,
candidate order, conditional prefix and legal masks retain version 1 semantics.
Compaction happens before packets, replay and learner batches are created.
Critic-only computations are omitted entirely for DQN. Role-inapplicable
component placeholders and common candidate widths remain unchanged.

The Q/actor first layer receives its own context, actor prefix and candidate.
The PPO value first layer receives only critic-visible context. It creates no
critic prefix GRU and no zero candidate/prefix input columns. Branch widths and
ordered layout are reconstructed from the encoder and checked in saved state;
algorithm participates in the observation identity. A version-1 checkpoint is
not silently converted into a version-2 model. New version-2 experiments need
fresh compatible initialization and retraining.

The shared batch entry point remains `smartsom batch-run`. It builds independent
algorithm contracts per entry; no equal-width network is required to mix PPO and
DQN in one queue. Existing `batch.stage` and `batch.requires` provide optional
DQN-before-PPO barriers, with resource admission within each stage. No learner,
reward, optimizer or scientific budget changes are bundled with compaction.
Smaller first layers do not by themselves establish an end-to-end speedup.
