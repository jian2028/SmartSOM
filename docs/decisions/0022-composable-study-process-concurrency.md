# 0022 — Composable study process concurrency and truthful progress

Date: 2026-09-26
Status: User-authorized implementation scope; validation recorded separately.

Supersedes ADR 0021's serial study execution restriction. Physical, policy,
sampling, seed, budget, validation and model package contracts remain unchanged.

Independent experiments run in bounded spawned processes. The Small authoring
recipe defaults to max_concurrent=8; one sequential environment and one CPU
numerical thread per experiment are retained. Legacy recipes without this field
retain serial execution. This follows the old PPO calibration's best tested
aggregate throughput configuration; no v3 performance claim follows.

The parent alone writes study state and aggregate progress. Workers write
separate state/log files. Shared paired controls have an exclusive file lock and
cached engineering failures remain failures unless explicitly retried. A failed
worker does not stop unrelated experiments. Interrupts propagate to live workers;
resume uses saved recovery snapshots, and completed experiments are skipped.

Runtime shows completed physical ticks, actual optimizer steps, policy-group
decision/sample counts, stage, case progress, active process count and each
worker's last activity. Training budget completion is not experiment completion.
Display-only reporting never consumes policy/environment RNG or changes inputs.
Old artifacts retain identity; implementation changes require new preparation.
