# 0026 — Retire legacy author formats

Date: 2026-09-28
Status: User-authorized decision; migration and verification are in progress.

Supersedes ADR 0025's compatibility promise for author-supplied v2/v3 Run,
Scenario, Composition, Policy and Study files. It does not change simulator
physics or retroactively change a historical result.

New executable experiments use Factory v2, Workload v3, Algorithm v2 and
Experiment v4. Workload v3 has mutually exclusive fixed Job, simple seeded
profile, and paired-template generators. A fixed Job retains stable Job and
operation IDs, per-machine durations, arrival/reveal, due date, priority and
rush metadata. The paired-template generator retains its distinct data streams,
shared Job pool and measured V contract. A simple profile or fixed list has no
inferred V condition. Faults remain an independent Factory extension outside H.

The four-file compiler consumes author files, then prepares detached physics,
Agent and execution objects. New frozen inputs use `smartsom.frozen-*` schemas
and an execution-config v2; a new subrun uses experiment v4. These are runtime
records, not additional author files. The compiler rejects older Workload YAML;
other legacy public entry points and engineering fixtures remain scheduled for
removal until their meaningful checks have moved. A source name, schema bump or
new runner cannot relabel old measurements as new-interface evidence.

The CLI and public Python API will accept only the four-file author contract
after migration. Old saved runs retain read-only log, metric and playback access,
but no training continuation or new evaluation through the current entry.
Historical reports and source commits retain their original commands and
results. Unsupported old author files are removed from the working tree only
after a per-file disposition and validation audit; Git history remains their
source of record.

This decision does not assert that the migration is complete, that old runs
have been converted, or that a new formal experiment has been launched.
