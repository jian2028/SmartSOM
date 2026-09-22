# 0018 — Multiple system input and output facilities

Date: 2026-09-22

Status: Accepted for the requested larger Studio factory templates.

## Decision

Factory authoring permits multiple `system_input` and `system_output` buffers.
This supersedes the single-facility-per-role validation rule documented in the
factory format under the original ADR 0014 authoring implementation. It preserves
ADR 0017's shared factory format and existing execution semantics.

Each facility retains its own semantic ID, geometry, storage capacity and explicit
ports. Multiple facilities are not aliases for a shared pool. The existing
production validator requires every demand to name `input_id` when several input
facilities exist, and rejects missing or non-input references. No new implicit
arrival distribution, output routing policy or workload generation is introduced.

## Evidence boundary

Templates 4–6 supply 8-, 20- and 40-machine layouts with distributed input/output,
inspection and scrap facilities. Geometry, connectivity, file and Studio checks establish
authoring behavior. They do not establish scheduling performance, traffic
capacity, deadlock freedom or a calibrated large-scale experiment.
