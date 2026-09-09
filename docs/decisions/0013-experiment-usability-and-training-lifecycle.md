# 0013 — Editable experiment recipes and a shared training lifecycle

Date: 2026-09-10

Status: Accepted design; implementation and platform qualification are staged.

The user approved a usability refactor with a common CLI and typed Python API.
An editable recipe composes reusable scenario/algorithm presets with local
overrides. Resolution validates and freezes a detached configuration before
execution. Every effective scientific value and override origin is recorded.
CLI and Python entry points use the same resolver and execution functions.

This supersedes the restrictions in ADR 0004/0009/0011 that scientific changes
must be authored only in algorithm files and that the CLI cannot override them.
The final resolved AlgorithmSpec still owns effective algorithm parameters.
Scenario and workload ownership, paired materialization and information visibility
remain unchanged. There is no second simulator or per-experiment hidden adapter.

The approved lifecycle extends the single-environment inference-only export in
ADR 0011/0012: update-boundary checkpoints include optimizer, algorithm dynamic
state, random state, sampler continuation and reconstructable active episodes.
Resume completes the original budget with the same source/configuration/runtime;
weight initialization creates a new experiment. Validation has independent inputs
and random state. Last and best are independent references, with immutable payloads.
Completed budget, early stopping, interruption and failure are distinct outcomes.

Best selection first compares completed-episode coverage on a fixed validation
set. Equal completion counts with different successful input sets are incomparable
and retain the incumbent; identical coverage compares makespan. Strict all-complete
and explicit custom objectives are supported separately. Failed episodes never
acquire a successful makespan.

Parallel sampling uses fixed global per-update quotas and stable logical environment
identities, independent of process IDs and completion order. Independent experiments
do not share trainers or evidence writers. Original single-environment seed recipes
remain available for historical fixed acceptance. New seed semantics are versioned.

Research extensions may replace public observation encoding, feedforward networks
and team/role learning rewards. They do not control semantic actions, masks,
coordination, environment termination or information access. Extension identity
includes implementation, parameters and numerical contracts; stateful extensions
must support checkpoint state. Raw physical rewards and research/learner rewards
are separate recorded quantities. Existing defaults retain their original meaning.

One experiment directory owns its real artifacts. External indexes and relative
shortcuts are rebuildable views. Historical files are read without mutation, and
exports carry real files with relocation metadata rather than dangling links.
Local evidence remains authoritative; cloud tracking is explicit and optional.

Learning batches and searches use a separate experiment identity with immutable
trial declarations and preserved attempts. This supersedes ADR 0009's restriction
on arbitrary parameter dimensions only for this new API; the v1 paired study
schema and its materialization contracts remain unchanged. Search objectives
use validation evidence, never held-out test outcomes. Early stopping, pruning,
execution failure and policy non-completion are recorded separately.

Historical acceptance reports retain their original source, meaning and failures.
The new final acceptance requires fresh runs from a clean integrated main commit.
This refactor requires macOS CPU qualification; Linux installation, H20 and CUDA execution
require independent later evidence. No performance advantage over SPT is required.
