# 0088 — Shared scientific calibration for search candidates

Status: user-authorized implementation; verification pending.

Extends ADR0087 for searches with `run.calibration`. Supersedes only the requirement
that every HPO candidate's changed algorithm match the template calibration identity.
Ordinary fixed experiments retain their existing strict calibration matching.
ADR0060's common physical windows, truncation and missing-metric rules remain intact.

Before launching a search, validate the original completed calibration against the
unchanged template. Every candidate then uses its validated fixed drain. Candidates
do not recalibrate and do not claim that the original report calibrated their changed
algorithm. No final test results determine this boundary.

Freeze the complete original report, its byte digest, the template's calibration
and scientific identities, the original reference, the shared drain and an explicit
derivation policy. Carry the report and derivation inside each candidate's frozen
inputs and portable Tune checkpoint. A changed report between checking and freezing
fails before execution. Recovery must validate frozen evidence rather than consult
the original external report.

Retain the exact calibration identity content too: original algorithm, factory,
workload, condition inputs and initial drain. Its bytes reproduce the original
calibration digest without the parent template directory. The template scientific
digest remains an explicit parent identity reference, not a claim that the candidate
has the same scientific configuration.

This choice keeps candidate horizons comparable. It does not alter Rule-based
calibration itself, ordinary fixed-run validation, evaluation multipliers, case
seeds, or the three unprioritized optimization objectives. It does not reintroduce
performance calibration. Engineering verification is not a research result.
