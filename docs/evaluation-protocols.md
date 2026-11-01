# Evaluation protocols and research evidence

Evaluation settings are scientific inputs in the Experiment file. They cannot be
changed by a launch profile or while resuming a frozen evaluation. `check` shows
the expanded protocol. Display/recording intervals are separate from scientific
measurement windows.

## Limited and completion evaluation

```yaml
evaluation:
  schema: smartsom.evaluation/v1
  mode: limited                 # explicit opt-in: completion
  common_window_end_tick: null   # last arrival + frozen drain_ticks
  completion_drain_multiplier: 5
```

Completion freezes a scenario through last arrival + multiplier × frozen drain.
Its orders and fault prefix agree with the limited scenario. It terminates when
all orders reach Output or at the guard. The original common window is retained.
An early fully drained case can contribute its final delivery count to the later
common-window denominator; interrupted or undrained cases cannot be extrapolated.
This does not extend resource-time integrals beyond actual observation.

`completion_protection_limit`, `drain_deadline`, and `wall_clock_interrupted` are
distinct. A wall-clock fragment is saved as `interrupted-case.json`, never as a
completed case receipt. Previously committed cases are not rerun on continuation.
The existing total wall budget is not reset by resuming.

`research-summary.json` contains repeat → world → condition equal-weight summaries,
coverage, and matched normal/low/medium/high throughput ratios. Complete cost J
requires all cases to have complete costs. Different maps and decoders must remain
separate. `evaluation_summary.across_seeds` reports sample SD (undefined for one
seed) and actual/planned counts, rather than treating cases as training seeds.

## Explicit controlled-recovery calibration

Recovery runs are separate `calibrate` / `evaluate` tasks. They retain the four
scientific input types; a candidate supplies an explicit evaluation workload
(orders, initial stock, interarrival time) derived from the referenced Workload.
There are no research defaults for supply, warmup, window size or tolerances.

Each `evaluation.recovery.candidates` item requires:

- `name`, `load: normal|saturated`, `orders`, `initial_orders`, `mean_interarrival`;
- `horizon_ticks`, `window_ticks`, `baseline_windows` (at least three);
- `minimum_window_deliveries`, `maximum_cv`, `maximum_relative_trend`;
- `faults`: explicit `machine_id`, `start_tick`, `duration_ticks` entries.

These are experimental parameters, not presets established by engineering tests.
The initial stock must be smaller than total orders. Supply must continue through
observation; an early end of arrivals does not establish a steady recovery result.

Set `evaluation.recovery.action: calibrate`, supply candidates, and run the normal
entry with `--task calibrate`. `run.calibration_cases` selects independent worlds.
The scan reports evidence; it never rewrites input YAML or silently chooses a
candidate. `recovery-calibration.json` records accepted candidates and identities.

The baseline uses disjoint windows immediately before the specified fault time.
Every window must meet minimum output; sample SD/mean must not exceed `maximum_cv`.
The absolute least-squares slope per window, divided by mean throughput, must not
exceed `maximum_relative_trend`. This is an explicit engineering validity criterion,
not a mathematical proof of stationary operation.

For saturated candidates, set `maximum_relative_capacity_gain`. At least one
otherwise comparable candidate with faster arrivals must also have a valid
baseline. Every matched method/world/repeat/fault baseline must show a small
relative output change, and observed baseline backlog must stay positive. Averaging
opposite changes across methods cannot establish saturation. The result describes
an empirical plateau over the tested supply range, not a universal capacity bound.

To evaluate, retain the same candidate definitions and methods, set:

```yaml
evaluation:
  recovery:
    action: evaluate
    selected: chosen-candidate-name
    calibration: path/to/recovery-calibration.json
    candidates: [] # replace with the explicit candidates used in calibration
```

The compiler freezes referenced report bytes and fingerprints; the empty candidate
list above is intentionally not runnable. A selected candidate must have accepted
calibration evidence. Learned policies must use the same frozen checkpoint as the
calibration. One paired normal/single-fault case is run per selected device, world,
method and inference repeat. Normal and disturbed cases share orders and policy
sampling seeds. Validation/test world counts follow their existing four-condition
case-count conventions (four ordinary condition cases represent one world).

After a detected drop below 95% of its own pre-fault baseline, three consecutive
non-overlapping windows at or above 95% confirm recovery. The first qualifying
window end and confirmation time are both saved. Adaptability is the reciprocal
of fault-to-first-qualifying time; repair-to-recovery is auxiliary and may be zero
when recovery preceded repair. No detected degradation, invalid baseline,
insufficient supply and failure to recover are distinct statuses, not zero or
infinite scores. Complete per-case evidence remains in `cases.jsonl` and
`recovery-summary.json`; do not report successful-only means as full-cohort results.

## Frozen Rule target and CAR

Produce a Rule validation `research-summary.json` with a complete J, then reference
it from the training Experiment:

```yaml
evaluation:
  learning_target:
    reference: path/to/rule-validation/research-summary.json
    cost_ratio: 1.0
    consecutive: 3
```

The compiler checks validation provenance, obtains Rule cost, and freezes the
reference file and checksum. An explicitly provided `rule_cost` must match it.
Actual validation receipts form the learning curve, separately by decoder. Missing
scheduled panels break persistence. `learning-target.json` records the target,
first qualifying episode, three-node confirmation and censored status when unmet.
CAR uses first qualifying episode after persistence is confirmed; it is a
common-target attainment ratio, not proof of mathematical convergence. Zero initial
attainment or incomplete pairs do not yield a finite ratio.

## SIG pairing and checkpoint provenance

SIG is `(J_Base - J_SL) / J_Base`; positive means lower cost. Aggregate complete
case costs within each training seed first, then form the relative improvement.
Also retain `J_SL - J_Base`. Missing complete costs or zero Base cost leave SIG
undefined.

Automatic pairing only permits the known built-in Base/Social observation and
encoder changes (`arm` and social `pool`). Actions, rewards, bindings, Mover,
learner settings, encoder width, Factory, Workload, training budget and evaluation
protocol remain part of the common signature. Cases additionally match world,
source, runtime, training seed, decoder and episode budget. Custom representations
need an explicit comparison definition; human labels are not sufficient evidence.

Independent checkpoint evaluation consumes a frozen `.pt.json` training-evidence
sidecar when available. Its checkpoint checksum and source identity must match.
Evaluation seed must never be substituted for an unknown training seed. Old
weights without that provenance can retain readable evaluation results but do not
automatically produce paired training-seed claims.

## Sampling and entities

Summary collection integrates per-machine, per-vehicle and all Buffer counters.
Detailed `entity_series` records actual physical boundaries at the configured
interval and the final boundary; it does not interpolate resource ownership into
unobserved ticks. Completion counters may use exact shipment timestamps between
those boundaries. Historical rows lacking these fields remain explicitly missing.

A CAR Rule reference also freezes the complete validation world inventory and its
panel definition (Factory, Workload, Rule decision protocol, traffic, evaluation
protocol and case count). Compilation rejects a different panel definition;
actual validation verifies the exact scenario hashes and implementation source.
The shared Rule target currently requires the deterministic built-in Mover, so
Rule repetitions and decoder labels do not change this target. Custom stochastic
traffic requires a separately specified reference protocol and is rejected here.
Three consecutive qualifying validation nodes are required. Validation model and
best-model copies retain only checksum-verified checkpoint provenance sidecars.
