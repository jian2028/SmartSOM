# 0100 — Rule and Random fixed panels through Tune

Status: user-authorized implementation; engineering verification recorded separately.

Supersedes only ADR0098/0099's deferral of named Base Rule/Random fixed-panel
evaluation. Learning checkpoint evaluation, centralized/Morris adaptation,
window/recovery analysis and new recordings remain deferred.

The maintained campaign has one Rule and one internal-only Random recipe per
size. Both share the learning campaign's scenario, protocol permissions, traffic,
fixed worlds/action seeds and inherited drain provenance. Direct legal semantic
candidates feed these non-learning policies; they do not select learned rule
aliases. Rule is the ADR0099 critical-ratio policy. Random uniformly samples its
legal candidates and is not a paper baseline. No optimizer or replay is created.

Public `check`/`run` accept `run.task: evaluate` with named `rule`/`random` methods
under Base control. Existing scientific case execution and atomic receipts are
reused inside the shared Tune Trainable. Ray Core case tasks execute within its
placement group; the coordinator and all workers are reserved up front. The
profile's evaluation count overrides experiment concurrency as for learning.
Evaluation worker CPU/memory declarations apply; GPU workers are rejected for
these CPU-only rules. No new scheduler, process pool or model saving is added.

Cases preserve frozen worlds, action seeds, conditions and full J/truncation
semantics. Non-learning evaluation reads the shared normalized task metrics before
telemetry/receipt publication; the historical simulator reward ledger and scientific
calibration behavior are not redefined. Receipts are the completion authority. Pause cancels uncommitted
work; Tune checkpoints carry committed receipts and published summaries, so
restore skips completed cases. Scientific tasks have no model best/last.
Historical runs, frozen calibration and source identities are not rewritten.

The retired normal-only Rule trace hashes are replaced by current legal-action,
completion and reward contracts. Dedicated reservation/credit tests remain.
Validate public parallel evaluation, deterministic seeded results, truncation,
checkpoint restoration and no duplicate results; keep missing platforms explicit.
