# 0047: Append-only manual queue around immutable experiment runs

Status: user-authorized implementation; engineering verification is separate.

Extends ADR0023–0025 for explicit live manual queues. Existing frozen directory
batch plans, strict source checks and automatic Tune scheduling remain unchanged.

An explicit live queue can adopt one existing V4 directory training batch. It
reads that batch's frozen plan, state and progress; the original driver keeps
running under its original source. Adoption never reloads weights, reinitializes
seeds, edits the original plan or changes a checkpoint's identity. The transition
therefore temporarily retains the original driver plus one queue coordinator.
Every adopted Experiment must have one entry and no dependency/evaluation gate.

Later independent stage-0 Experiments are checked and frozen individually before
an append-only add request becomes visible. Added entries retain their scientific
identities and run with native manual execution, one environment/thread layout as
authored, and an independent training process. They share the queue's admission
limit. Duplicate file-stem task IDs and changed frozen inputs are rejected.
Queue mutation commands use an atomic request journal; only the driver writes
applied status. The original frozen queue manifest remains immutable.

The manual limit is positive and adjustable through CLI or controlling attach
(`c`, digits, Enter). Raising it admits waiting jobs; lowering it never kills,
pauses or restarts active learners. Existing workers drain naturally. The queue
checks actual available memory before starting additional workers rather than
reserving nearly all memory for each unknown calibration group. This is an
operator-selected concurrency limit, not automatic throughput optimization.

The queue remains open when idle. Explicit close prevents later additions and
finishes after already accepted work drains. Cooperative stop propagates through
verified original and new ownership protocols; no unrelated process is killed.
Monitor remains read-only. Only attach accepts concurrency changes.

The dashboard shows host CPU busy logical cores / available logical cores,
available memory, running / requested concurrency and rolling aggregate execution
physical ticks per wall second, with training throughput separately reported.
Execution includes observed validation/evaluation ticks; polling may omit a
case-ending tail when the next case starts between samples. The throughput
window includes checkpoint waits and excludes newly observed historical ticks, and is not a claim
of an algorithm or hardware performance gain. Each original/new child keeps its
own source and scientific provenance. Queue orchestration is development evidence
until the relevant platform and source gates have passed.
