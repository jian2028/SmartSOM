# Small-map 512-job machine-outage pilot

Local engineering pilot, 2026-10-24. This records the current controller
checkpoint and the next identified improvement; it is not a formal experiment,
milestone acceptance or replicated performance claim.

## Frozen design

- Source: `c5fd3769e50cf6f114f15e6f3f54719e6763d82a`.
- Template 7: eight machines and eight AGVs.
- 512 original orders per case, all released at tick zero, following
  operation 1 → 2 → 3 → 4. Nominal processing is one tick per operation,
  due date is tick 1,000, and priority/input location are shared.
- World seed 202; Machine/Buffer/Dispatcher/Mover seeds 11/12/13/14.
- Rules: `normal_first`, `edd`, `nearest` with traffic admission and work in
  progress first, and `clearance_shortest_path`; first-arrival pickup matching.
- Matched control without an outage; disturbed case with Machine 2 unavailable
  on the half-open interval `[2000, 4000)`.
- Compare normal, unavailable and restored phases through tick 6,000, then
  continue both cases to all-demand shipment, with a 12,000-tick safety limit.

Machine 2 was selected before execution using the highest prebuffer delivery
count in the preceding 256-job run (271). Both cases use identical frozen
inputs except for the outage. Neither case resets at the phase boundaries.

## Results

Throughput counts unique original orders delivered per 1,000 physical ticks.
The initial interval includes warm-up.

| Interval | Control deliveries | Outage deliveries | Control rate | Outage rate |
| --- | ---: | ---: | ---: | ---: |
| 0–2,000 | 116 | 116 | 58 | 58 |
| 2,000–4,000 | 122 | 108 | 61 | 54 |
| 4,000–6,000 | 120 | 118 | 60 | 59 |

Output during the outage decreased 11.48% relative to the same control interval.
After repair, the phase-average rate was 1.67% below the control. A 14-order
delivery deficit existed at repair; 16 remained at tick 6,000. The maximum
primary-window deficit was 23 at tick 5,429.

Both cases delivered all 512 original orders. The control finished at tick
8,560 and the outage case at 8,787: 227 ticks later, or 2.65%. Mean paired order
delivery delay was 189.14 ticks; the median was 213.5 and the maximum 1,989.
There were 387 later, seven earlier and 118 unchanged order deliveries.
Both cases had zero rejected AGV actions, persistent reversals and detected
longer cycles, and 512 qualified final deliveries in privileged evaluation data.

The earliest qualifying near-level intervals for the outage case were:

| Phase | Interval | Mean 500-tick rolling rate |
| --- | --- | ---: |
| Normal | 550–1,550 | 60.57 |
| Unavailable | 2,500–3,500 | 54.19 |
| Restored | 4,800–5,800 | 56.95 |

The descriptive rule requires a positive 1,000-tick block, 50-tick samples,
coefficient of variation and fitted relative drift at most 10%, fully
phase-contained 500-tick windows, and at least 25% of orders still unshipped at
confirmation. These are retrospective intervals, not permanent steady states.
No interval through tick 6,000 met the separate strict recovery rule of staying
within ±10% of the simultaneous control's 500-tick rate for 500 consecutive
ticks. Average-rate recovery and delivery-debt recovery are distinct outcomes.

## Controller finding and next work

Machine 7 maintained operation-2 production while Machine 2 was unavailable:
113 starts versus the control's 134 starts on Machine 2 during the outage.
However, four jobs entered Machine 2's prebuffer at ticks 2,008, 2,023, 2,030
and 2,031, then waited 1,969–1,992 ticks until repair. None was already targeting
that prebuffer in the shutdown snapshot. Prebuffer occupancy averaged 3.954
jobs versus 0.069 in the control. The controller redirects later work when the
buffer fills but does not proactively avoid admissions to the stopped machine.

The next controller change should use current machine availability to avoid
those admissions when a working alternative exists, and reconsider inbound
destinations while preserving service commitments and reservations. Existing
queued jobs must follow the environment's movement contracts rather than being
relocated through direct state edits. Repeat the frozen pair after the change,
then repeat across paired seeds and varied workloads before generalizing.

Machine processing alone was four ticks for each successful final attempt in
both cases. Including work on scrapped attempts, the control used 2,195 machine
ticks (4.287 per delivered order), and the outage case used 2,190 (4.277).
These exclude transport, waiting, inspection and downtime. One-tick operations
do not exercise interrupted multi-tick processing and resumption.

## Evidence and limits

The local evidence remains outside Git in
`E:\USC\IMPACT\Small Map 512 Machine Outage Pilot 2026-10-24`:
frozen scenarios, source/policy identities, commands, events, per-tick resource
and shipment data, individual delivery times, state snapshots, analysis, and
four PNG/SVG figures. The original generated files and large replay trace are
not included in this checkpoint.

- `plan.json` SHA-256:
  `e8e80c46754b1c2a583f6827d53f16fa7b6427072b316764aa81f569b4eabc61`.
- `analysis.json` SHA-256:
  `73304384a4410788ae40d8062197023aa04a9b34c835b5de45cfcd0851d1279b`.
- Pre-outage telemetry and commands matched; every shipment count matched
  individual shipment times. The machine was unavailable for exactly 2,000
  physical intervals and started no processing while down. Core/rule source
  remained unchanged during execution.
- A compact evidence adapter omitted duplicated policy-observation history.
  Before use, physical results and controller state matched the normal driver
  on a complete four-job outage case and a 512-job prefix. It did not change
  policy choices or physical transition rules.
- A native visual replay was subsequently reconstructed from the saved
  commands. All 91,496 original events, 17 saved 500-tick state hashes, the phase
  snapshots, initial/final states, and all shipment times matched. Original
  full decision-observation histories were not recorded and are not audited.

This is one paired seed, a fixed initial batch and local Windows engineering
evidence from the identified feature-branch source. It does not establish
statistical repeatability, broader map/workload behavior, AGV-failure response,
or full Linux/macOS qualification. A formal experiment still requires an
integrated `main` commit or explicit tag under `CONTRIBUTING.md`.
