# 0031 — V4 Algorithm matrix axis and per-entry best ranking

Date: 2026-09-29
Status: Accepted for development experiments

ADR 0025 established four author files and the Experiment V4 compiler. This decision extends V4 in place: an Experiment may use one top-level `algorithm` or a nonempty `matrix.algorithms` list, but not both. The full Cartesian product is Factory × Workload × Algorithm × training seed. Each expanded entry freezes exactly one Algorithm and has an independent training, checkpoint, validation, and held-out evaluation lifecycle. The single-Algorithm interface and its identity remain compatible.

`--algorithm` replaces the Algorithm axis for one `check` or `run`; `--set algorithm.*` must validate for every selected Algorithm. Shared Factory, Workload, `data_seed`, and case index produce paired physical cases across Algorithms. Invalid combinations fail before execution. Internal entry IDs remain stable; the display also names the Algorithm file and seed.

The optional `completion_delivery_return` validation mode ranks an entry's updates by completed frozen cases, average qualified deliveries over all frozen cases, then average raw return over all frozen cases. An exact tie keeps the earlier update. Engineering failure, missing cases, or nonfinite deliveries/return makes that update ineligible. With no eligible update, `best` evaluation fails rather than substituting `last`. The old selection modes remain available.

Batch 02's four pending PPO training configurations are represented by one V4 matrix with four Algorithms and two seeds. This decision does not add a cross-entry winner, rule-control gate, resource calibration, capacity probe, or ten-hour batch deadline. Those stages remain separate. Development runs from an uncommitted main checkout are not formal research evidence under the repository's source-identity rule.
