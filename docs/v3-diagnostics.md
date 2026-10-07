# V3 diagnostic and provenance contracts

These records describe execution and learning evidence. They do not select an
action, change a reward, certify useful behavior, or alter physical feasibility.
Historical experiment files and frozen manifests must retain their original
schemas and hashes; do not backfill new fields as if they had been measured.

## Pickup observations

`dispatcher_diagnostics` uses `smartsom.pickup-diagnostics/v2`. Native evaluation
collects one bounded summary per case. Native training maintains a separate
summary per environment episode, outside simulator snapshots and observations;
the summaries are saved at checkpoint boundaries and attached to completed
episode rows. Current episode summaries appear in the training run record.
Restoring an old checkpoint starts a partial observation interval marked
`history_complete=false`; counts are observed since restoration, not lifetime
totals. A new episode starts complete accounting.

| Field | Numerator or meaning | Denominator and phase |
| --- | --- | --- |
| `empty_target_decisions` | Empty AGV target selections | Actual Dispatcher requests; excludes travel ticks without a request |
| `ready_source_choice_opportunities` | Requests with at least one legal target whose source currently has ready jobs | Empty target decisions, using that request's detached observation |
| `ready_source_choices` | Such requests selecting a currently ready source | `ready_source_choice_opportunities` |
| `nonready_source_choices_when_ready_available` | Such requests selecting another source | Same denominator; a preference, not an error |
| `ready_source_choice_fraction` | Ready choices / opportunities | Null with zero opportunities |
| `admitted_service_slots` | Sum of actual Buffer prefix lengths requested | Source-local post-admission service phase; each prefix counted once |
| `service_opportunity_boundaries` | Boundaries with a nonempty admitted Buffer request | Observed boundaries; not a counterfactual assignment optimum |
| `pickup_started` | Committed pickup-start events | Completed service starts, not target selections |
| `arrival_to_pickup_ticks_*` | Event tick minus recorded physical arrival | Only starts with a known arrival; count, sum, max and mean retained |

Latency is conditional on pickups that started. It is not an uncensored mean
over all waiting vehicles. Travel, lack of ready work, port competition and
earlier FIFO service can explain ticks without pickup. No metric here labels
every such tick a missed action. Storage is constant per environment episode.

The old `eligible_pickup_boundaries`, `eligible_empty_agv_decisions`,
`missed_pickup_boundaries` and `first_reservation_tick` are null with an explicit
legacy-status explanation under the current nonexclusive-intention contract.
Their old definitions are not reused. The historical NO_REQUEST helper and its
optional reward-shaping caller are unchanged. Configured `global_optimal` is
recorded separately from the effective source-local `first_arrival` semantics.

## Learning coverage

Native learner summaries use `smartsom.native-learner-diagnostics/v2` and expose
a `coverage` object for each policy group. PPO separates effective actor sample
exposures from critic/value exposures. DQN separates TD training exposures with
multiple legal alternatives, forced single alternatives, and unknown historical
replay attribution. Exposures count replay reuse or repeated PPO epochs, not
unique experiences. Collector choice/forced counts describe requests, not
optimizer minibatches. New tags are observational only; replay sampling,
priorities, targets and loss formulas do not consume them.

An optimizer count greater than zero proves an update ran, not that every actor
had effective decision samples, that its gradient was nonzero, or that behavior
improved. Shared-group samples cannot be attributed to each owner. Older replay
rows lack choice tags and are explicitly unknown; restoring old collector state
marks new decision counts incomplete. Old diagnostics remain readable and absent
historical values are not synthesized as zero. The existing resume state carries
these counters separately from learning and simulator state.

## Effective model identity

New model packages carry `effective_model_contract` and its SHA-256. It identifies
the actual encoder and network classes and semantic versions, dimensions,
projection, observation extension, physical-job inspection identity, declared
actor/critic/Q branches, conditional prefix GRU width and scope, and critic-only
context masking. The GRU encodes a within-boundary selection prefix; it is not an
across-tick recurrent policy. Actor and Q inputs omit the eight physical-job
critic summary fields; public global resource state is still available.

New packages validate declared semantic identity before weight deserialization
and validate the constructed effective contract before applying weights. New
continuation states validate the effective contract before mutating session
state. Historical packages/checkpoints without the new contract retain existing
validation; absence is not represented as verified new provenance. Identity
recording does not enable or install the physical-job encoder automatically.

Native training run records, component exports, continuation state and evaluation
case manifests retain effective model identities. Public API and V4 author/batch
paths use the same lifecycle; a local report-only patch is not needed for native
behavior. Source-tree identity remains a separate record from portable semantic
identity, so platform-specific paths are not part of semantic hashes.

## Explicit caller or harness provenance

Local drivers can use the thin standard-library context below around preparation
and native execution. It freezes portable file labels and their SHA-256 hashes in
`PreparedComposition.execution_provenance_json`; native worker serialization and
run/case records retain it. This extra provenance does not change the scientific
configuration digest. Without a declaration, `caller_status=not_declared`, not an
invented native-only certification. No filenames are discovered heuristically.

```python
from smartsom.learning.production_provenance import execution_provenance

with execution_provenance("batch04", {"scripts/driver.py": __file__}):
    from smartsom.config.experiment import prepare

    prepared = prepare(config)
    result = api.train_prepared(prepared)
```

An existing completed Batch04 driver and its frozen files must not be rewritten.
A future explicitly admitted driver can use this context instead of duplicating
native metadata. Its full harness file set should be declared, not just the
entrypoint. Scientific protocol identity, active model contract and execution
harness/source identity are complementary evidence, not interchangeable hashes.


## Additional bounded analysis

`ModelPolicy` observes its existing forward output without a second forward or
random draw. Group summaries distinguish legal candidate count, top-two gap and
exact ties, selected-vs-greedy, PPO categorical probability/entropy, and actual
DQN epsilon and exploratory-branch selection. Epsilon-greedy behavior probability
includes exploration that happens to choose the greedy action. The legacy DQN
`log_probability` remains categorical-over-Q for compatibility and is explicitly
not described as behavior probability. `request.deterministic` means one legal
candidate, not evaluation mode. Scores are stratified by role and AGV load/motion/
arrival state. At most eight capped-identity examples are retained per group.
Their identities describe semantic targets; there are no target-sized histograms.

Pickup observers additionally retain separate pre-proposal no-ready-stock,
loaded-destination-full and retargeting counts; post-commit processing, travel,
service and co-located shared-port counts; and pickup/operation/shipment progress.
Each ratio has its own phase-specific denominator and unknown/not-applicable
counts. These conditions can overlap and do not establish counterfactual blame.
A shared port is competition evidence, not proof a policy missed an opportunity.
Processing counts are status observations, including a processing job paused by
an outage; they are not a replacement for committed productive-time metrics.

An absent progress event for 128 observed episode ticks or a committed rejection
triggers a compact event-context snippet. Each episode retains at most eight
snippets, at most 256 KiB encoded snippet payload inside a 512 KiB event budget,
four recent boundaries, and eight compact decisions/events per boundary with
64-character identifiers. Omitted event/decision counts and
suppressed triggers remain explicit. Snippets are ordered context, not full
replays or proof of causal attribution. Open progress gaps are right-censored;
whole observed gaps and since-start gaps are descriptive, not service-latency
estimates. These counters and rings survive checkpoints outside simulator state.

PPO reports actual clip fraction, raw pre-normalization advantage mean/absolute
mean and pooled value explained variance. DQN reports Q, target and TD-error
means/absolute means/mean squares, physical transition dt, terminal fraction and
replay age in **insertions**, using only the minibatch already selected for the
optimizer. Unknown historical replay ages are excluded from the observed weight.
All tensor statistics detach; neither gradients, targets, sampling nor updates
consume diagnostics. Constant or unobserved value targets yield unavailable EV,
never a fabricated zero. Metric weights are repeated training exposures. Summary
weighted sums permit correct interval subtraction; old means and weights can also
be consumed with their original floating-point precision.

Native `finish()` writes `reports/episodes.json` with whole simulator episodes and
partial budget episodes separately, plus `reports/learner-intervals.json`. Dev
cases remain separate validation files. Cohort assignments come only from frozen
workload provenance, with common/novel declared and released denominators, shipped
and qualified counts, and an explicit unknown-assignment count. No ID/route guessing
is used. Passing rates condition on shipment; unreleased jobs remain in declared
cohort denominators. Missing historical assignments or outcomes remain unknown.

A reusable read-only consumer creates a new derived JSON report:

```sh
uv run python -m smartsom.experiments.diagnostic_report --run RUN --baseline BASELINE --output NEW_REPORT.json
uv run python -m smartsom.experiments.diagnostic_report --layout batch04 --run CELL --baseline RULES --output NEW_REPORT.json
```

The native reader accepts a case suite or a run's `evidence/` / `evaluation/` case
folders. The thin Batch04 adapter verifies committed result file hashes, reads
whole training windows, dev suites and weighted learner history, and never loads
checkpoint objects or edits historical raw results/protocols. Paired differences
use case/replication/seed identity; matching world hashes verify the pairing,
missing hashes are explicitly unverified, and conflicting hashes suppress deltas.
Duplicate suite keys fail rather than silently pool checkpoints. Empty evaluation
is flagged as a failure. Shipment, passing and tardiness reward contributions use
the recorded contract and denominators. The zero-shipment passing contribution is
zero only because that reward contract defines it so; passing-rate metrics remain
unavailable. Standalone exports are capped at 16 MiB; bulky examples are explicitly omitted
first, and an oversized remaining report is rejected. Standalone reports use
exclusive creation and cannot overwrite existing files. The export cap applies to each standalone export, not native finish reports,
a whole run or a collection of experiment cells. Live observer retention is bounded
separately. Episode rows, raw update history, actions, checkpoints and logs can grow
with run length; full replays and Ray copies add further storage.


## Diagnostic recording configuration

All diagnostics remain on. An optional experiment-level section controls only
bounded event recording and derived report granularity, in both V3 and V4 YAML:

```yaml
diagnostics:
  event_context:
    max_snippets: 8
    max_payload_bytes: 262144
    lookback_boundaries: 4
    stagnation_ticks: 128
  reports:
    interval_updates: 1
```

These are the defaults for omitted fields and historical configurations. Unknown
keys, booleans, strings and noninteger values are rejected. Accepted ranges are
1¨C64 snippets, 4096¨C1048576 payload bytes, 1¨C32 retained boundaries, 1¨C2147483647
physical stagnation ticks, and 1¨C1048576 updates per derived report interval.
There is no diagnostic enable flag or off/light/full profile, and no setting
changes observations, rewards, actions, optimizer updates or minibatch selection.
The payload limit bounds encoded retained snippets; it is not a whole-run limit.

Preparation freezes the effective defaults and overrides in the run configuration.
Native training, evaluation and directory batch workers consume that configuration.
Execution provenance records `smartsom.diagnostic-capture/v1`, normalized settings
and their SHA-256 in run records, model packages, continuation states and cases.
This observational identity is separate from scientific configuration and model
semantic identities; changing it does not invalidate otherwise compatible weights.
Each raw update also records the capture identity active for that update.

Public resume uses the frozen run configuration. A caller explicitly preparing
compatible settings before restoring a continuation state retains the learning
and physical state exactly. If event-context settings change, prior counters are
preserved but retained context examples and the recent ring are cleared. The
transition records the effective tick, old/new hashes and discarded counts;
the last eight transitions are retained with an omitted-transition count. Context
history is marked incomplete, and the new threshold starts at the transition,
without retroactive triggers. Unchanged settings preserve the context exactly.
Missing historical measurements remain unavailable rather than synthesized zeros.

`reports.interval_updates` groups the exported learner intervals by consecutive
updates using cumulative weighted sums and observed weights. Intervals split at
capture identity changes and include their first/last update and observed count.
Raw update history remains intact, and the last partial interval is retained.
The standalone native report reader honors the frozen configuration; historical
Batch04 reports use one interval per update because no such setting was captured.
