# Registered observation fields (local preparation)

The new encoding is opt-in. Keep the existing four-file workflow and public
`smartsom check`/`batch-run` commands. No experiment was launched to qualify this
preparation. [ADR0041](decisions/0041-registered-public-observation-fields.md)
records compatibility and the explicitly selected scope.

Select both implementations in each new-model policy. The lists below are an
interface example, not an approved Batch05 feature set:

```yaml
kind: new_model
extensions:
  observation:
    name: builtin.public_fields
    version: "1"
    parameters:
      rounding: ceil
      include_inspection: true
      context:
        - {name: decision, normalization: projection, aggregation: per_entity, visibility: actor}
        - {name: self_target, normalization: raw, aggregation: per_entity, visibility: actor}
        - {name: local_map, normalization: raw, aggregation: per_entity, visibility: actor, applies_to: [dispatcher, mover]}
        - {name: vehicles, normalization: projection, aggregation: per_entity, visibility: both}
        - {name: machines, normalization: projection, aggregation: per_entity, visibility: both}
        - {name: inventory, normalization: capacity, aggregation: per_entity, visibility: both}
        - {name: work_summary, normalization: projection, aggregation: per_entity, visibility: critic}
      candidates:
        - {name: job, normalization: projection, aggregation: per_entity, visibility: actor}
        - {name: semantics, normalization: projection, aggregation: per_entity, visibility: actor}
        - {name: machine_mode, normalization: projection, aggregation: per_entity, visibility: actor}
        - {name: dispatch_target, normalization: capacity, aggregation: per_entity, visibility: actor}
        - {name: cargo, normalization: projection, aggregation: per_entity, visibility: actor}
        - {name: movement, normalization: projection, aggregation: per_entity, visibility: actor}
  network_implementation: {name: builtin.scoped_candidate, version: "1"}
  network:
    actor: {hidden_sizes: [128, 128]}
    critic: {hidden_sizes: [128, 128]}
```

Other AGVs' has-target flags and semantic targets are excluded from both branches.
No component exposes target counts or counts divided by capacity. Own target
remains in `self_target` and Dispatcher rows. `attempt` is recorded in core
counters and diagnostic state and excluded from job, prefix and cargo inputs.

| Component | Fields and order | Normalization / aggregation |
| --- | --- | --- |
| decision | physical occupancy, unlimited flag, four role flags, request count, prefix count, own cargo/service/feedback | projection or raw; per_entity |
| self_target | target presence, own cargo, owner and port one-hot | raw; per_entity |
| local_map | applicability bit, then 5×5 pairs of obstacle/outside and other physical AGV occupancy, centered on owner AGV | raw; per_entity; Dispatcher/Mover only |
| vehicles | x, y, cargo presence, service presence, sorted by AGV identity | projection or raw; per_entity/mean/max/sum |
| machines | down, IDLE/READY/PROCESSING/BLOCKED, public nominal, elapsed, job presence | projection or raw; per_entity/mean/max/sum |
| inventory | occupancy, ready, prospective supply, capacity, unlimited, PASS/UNKNOWN/FAIL, DISPOSING, available next-operation counts | capacity/projection/raw; per_entity/mean/max/sum |
| topology | static solid-cell bitmap | raw; per_entity |
| work_summary | urgency min/mean/negative fraction, remaining mean/max/pressure, remaining fraction mean, visible-work presence | projection or raw; critic only |
| job / cargo | remaining fraction, urgency, work, priority, age, quality presence/PASS, risk/presence, remaining steps, job presence | projection or raw; actor only |
| semantics | operation one-hot/counts, location/port one-hot, capability and normal nominal by machine | projection or raw; actor only |
| machine_mode | selected nominal, mode time scale, public error rate/presence, normal-mode flag | projection or raw; actor only |
| dispatch_target | presence, distance, x/y, loaded, supply, physical occupancy, capacity, unlimited, full, own-current-target, ready | capacity; actor only |
| movement | existing legal movement geometry/physical occupancy tuple with declared time scaling | projection or raw; actor only |

Per-entity order uses stable sorted resource identities. Aggregation applies
independently to each column; mean/max of an empty resource set is zero. Context
layout stores the ordered component definitions and widths. Candidate/prefix
width is shared across roles; components inapplicable to a role are zero.
`applies_to` can explicitly narrow applicable roles. The local_map component is
intrinsically Dispatcher/Mover-only; Machine/Buffer receive 51 zeros rather than
a neighborhood centered on (0,0). The leading bit is one for an applicable AGV
owner. This keeps shapes equal while making absence distinguishable from empty
geometry. Applicability and field order participate in frozen package identity.
Machine semantics refer to the route after the proposed START; conditional Buffer
prefixes refer to the selected job's current route, as in the original contract.

Projection divides times/counts by the existing positive `time_scale` and
`count_scale`; coordinates divide by map width/height. Job priority retains
the established /10 scaling in both modes. Remaining work fractions and
boolean fields remain dimensionless. `decision` uses finite global physical
capacity for occupancy, or count_scale when any included owner is unlimited;
`include_inspection` chooses inspection participation in that global denominator.
In inventory and target rows, capacity normalization divides finite counts by
capacity. For unlimited owners, count_scale plus the unlimited flag explicitly
distinguishes the fallback from a finite-capacity ratio. End-sink occupancy,
ready and supply are always zero. No clipping hides over-capacity states.

Public nominal work averages rounded normal-mode durations over capable
machines. In-flight remaining reference work uses public elapsed time relative
to the declared current-mode nominal duration, bounded to [0,1]. It does not
read realized remaining time. Hidden risk/error rates use zero plus a presence
bit; UNKNOWN quality remains present with PASS zero. No hidden normalization
statistics are updated during inference or loaded from another field layout.

AGV positions in travel-time-matrix mode permit abstract road overlap; this
encoding does not imply road congestion or collision enforcement. Port service
competition still applies. A future learned physical Mover must explicitly use
grid transport. The Dispatcher→Mover semantic target interface stays unchanged;
this preparation introduces no Mover algorithm or further spatial deletions.

The existing selector, checkpoints, nonuniform dev schedule, bounded retention,
stopping/recovery and online concurrency paths remain the integration points.
The local tests cover compilation and inference package round trips for PPO/DQN;
native learning, spawned sampling, lifecycle and full CI remain pending the
environment and main-CI gate.
