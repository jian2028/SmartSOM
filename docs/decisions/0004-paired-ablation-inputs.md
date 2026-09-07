# 0004 — Preserve unchanged inputs in paired ablations

Date: 2026-09-07

Status: Accepted design contract; study/batch execution is not implemented.

## Context

ADR 0002 pairs algorithms against one materialized world. Its non-algorithm
parameter-cell seed rule needs refinement for environment-module ablations:
including an on/off switch in that shared seed identity would also regenerate
the workload and other disturbances. The comparison would change more than the
declared factor.

## Decision

For future paired studies, a compatible base case and replication identify the
shared inputs. Algorithm variants and declared module-ablation switches do not
change the seeds of unchanged input components. Materialize each required shared
component once and give child runs immutable inputs through the ordinary
`ResolvedRun` / `run_one()` path.

For algorithm comparisons, all world-component digests must match. For module
ablations, compare the unchanged component digests and separately record declared
disabled or modified components. An ablation cannot pass a pairing check by
silently ignoring arbitrary world differences. Factory or workload structural
changes form distinct base cases; identical numeric seeds alone do not make
their realized inputs identical.

Standalone runs retain root seed ownership in `run.yaml` and the exact current
`smartsom.seed/v1` recipe. A future study definition (the `BatchSpec` in ADR 0002)
owns its root seed; children record derived effective seeds. Algorithm/solver
randomness remains separate from world inputs. Imported inputs remain fixed,
with historical seeds serving only as provenance.

The scenario already represents a reusable case. Do not add a separate case file
or per-run adapter file. Algorithm providers and their scientific parameters
belong in algorithm presets. Shell launchers may select execution resources and
operational settings, but must not hide a second scientific configuration.

## Compatibility and limits

This supersedes only the interpretation of the batch parameter-cell identity for
declared paired ablations in ADR 0002. Existing algorithm-only pairing and all
single-run inputs, seeds, schemas and evidence remain unchanged. Exact study
schema, batch seed encoding, resume identities and concurrency are deferred to
the first actual batch implementation. Current tests establish single-run module
independence; they do not establish materialize-once batch support.
