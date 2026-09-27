# Factory: layout, H and breakdowns

Existing v2 Factory files need no rewrite. Reliability is an optional top-level
field alongside `factory` and `authoring`; this fragment shows its shape:

```yaml
reliability:
  enabled: true
  defaults:
    enabled: true
    uptime:
      distribution: uniform
      min_ticks: 300
      max_ticks: 600
    repair:
      min_ticks: 10
      max_ticks: 30
  machines:
    machine_001:
      enabled: false
```

Replace the uptime object with `distribution: exponential` and `mean_ticks`
for exponential available-time sampling. Overrides use stable Machine IDs and
inherit unchanged default fields. Unknown IDs, reversed ranges or incomplete
active rules fail validation. A global `enabled: false` disables faults.

Uptime includes idle time while a machine is normally available. Repair does not
sample another failure. A failure pauses processing, retaining the Job, selected
mode and completed work; repair resumes the same operation. The run freezes the
realized half-open outage intervals. A Factory stores the sampling rules, never
the generated event schedule. Studio preserves these fields when reading and
saving; this change does not add a reliability editor panel.

H reports speed, quality and combined components from the declared machine table.
Breakdowns are an independent factor and do not enter H. For paired H conditions
of one scale, checks compare geometry/resource identities, operation compatibility
and nominal per-operation/per-mode capacity and capacity-weighted quality. A
legacy Factory without the required three-mode contract receives a diagnostic
rather than a fabricated H value. Nominal equality does not imply equal realized
throughput or eliminate duration-rounding differences.
