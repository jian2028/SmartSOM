# 0009 — Paired studies and restartable single-host execution

Status: Accepted and implemented.

This implements the study subset of ADR 0002 and the pairing rule of ADR 0004.
It narrows the earlier general sweep/override design: v1 accepts explicit case,
algorithm, replication and module-disable Cartesian dimensions, with no arbitrary
parameter paths, zip mode, scientific CLI overrides or dynamic provider imports.

A scenario is a case. `smartsom.study/v1` owns the root seed and references cases
and algorithm presets by stable IDs. Cases, algorithms and variants are expanded
in ID order, replications in numeric order; presentation/input order does not
change identity. Invalid combinations fail before output allocation. Disabling
an inactive module is an error; incompatible combinations are not silently
repaired. In particular a fixed-quality policy requires enabled quality, and
arrival-event decisions require arrivals. Use compatible scenarios/variants.

World root = first eight SHA-256 bytes, big endian, of canonical compact UTF-8 JSON
`["smartsom.study-world/v1", study_seed, case_id, replication]`.
Algorithm root uses `["smartsom.study-algorithm/v1", world_root, algorithm_id]`.
Existing seed/v1 derives world domains from the world root, algorithm/solver
from the algorithm root. Standalone seed/v1 is unchanged. A child's explicit
study seed origin explains the mixed derivation; ordinary run YAML admits no
named-seed overrides. Disabled modules do not enter unchanged components' keys.

Prepare each case/replication once, then bind algorithms and variants to immutable
inputs. Disabling UPT re-expands quality durations against nominal base times
using the same quality draws. Factory/workload structural changes are separate
cases; numeric seed equality alone is not a pairing guarantee.

`run_batch(ResolvedStudy)` uses bounded spawn processes (default two). Each worker
calls the existing `run_one`; only progress and compact results cross back to the
coordinator. No engine checkpoint or alternative state machine is introduced.
All plan entries are currently held in memory and full traces remain in workers;
this is not a claim of bounded-memory arbitrary-scale study execution.

A study saves a checksummed immutable plan and full child input snapshots. Input
paths are historical provenance; snapshot reload never follows them. Resume
requires the same code commit, executable source digest, Python and dependency
versions and verifies completed artifacts and scientific identity. Preserve all
attempts; reuse verified successes, restart incomplete attempts, and retry failed
attempts only explicitly. Per-study and per-child OS locks exclude active owners.
First Ctrl+C drains active runs without starting more; second interrupts only
owned workers, retaining partial evidence. Abruptly lost processes are incomplete,
not successful. No automatic cancellation/rerouting or deadlock recovery occurs.

Study recording defaults to observation hashes; explicit full recording and the
existing standalone default remain available. Hashes are separately named and
never presented as full observations. Semantic trace and realized inputs always
remain. Progress is human/operational evidence; opt-in debug is bounded to three
10 MiB files and does not duplicate snapshots. Neither changes scientific identity.

macOS is tested locally. The same subprocess tests are collected by Ubuntu CI;
Linux success requires a real completed CI result, not just the presence of YAML.
