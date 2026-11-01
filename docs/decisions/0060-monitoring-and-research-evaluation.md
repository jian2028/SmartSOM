# 0060 — Explicit monitoring and research evaluation protocols

Status: accepted; implementation and final acceptance are tracked separately.

Supersedes the monitoring presentation and evaluation protocol portions of
ADR0058. ADR0059's component, action and checkpoint contracts remain unchanged.
This decision authorizes engineering acceptance, not formal research runs or
claims that one learning method is superior.

## Evidence and presentation

Local events remain authoritative. Training defaults to bounded summary evidence;
validation and final evaluation collect detailed evidence. Training detail is
explicit. Resource summaries include every machine, vehicle and Buffer, including
Input/Output. Detailed local evidence does not imply unrestricted cloud uploads.
Progress and cloud training updates default to five wall seconds, physical
detail to fifty ticks. Exact metrics must not depend on plotting cadence.

Foreground, attach, directory batches and Tune consume the same progress facts.
Committed training counters remain separate from in-flight episode counters.
An episode budget uses episodes; physical budgets use ticks; case stages use
cases. Unknown budgets/ETAs remain unknown. A pause request is not a durable pause.
Program execution speed and production throughput have separate units and names.

Each seed/trial owns one tracking run, with an explicit or launch-unique group
retained across continuation. Names and free labels aid navigation but never
replace scientific identities. Resolved scientific inputs and source/component
identities are recorded, while credentials, entire repositories and large model
files are not automatically uploaded. Cloud failures do not invalidate local
commits, and observers cannot consume policy RNG.

Dashboard creation is explicit and creates new saved views. Small/Medium and
sample/argmax remain separate. Each of delivery coverage, drained-case fraction,
complete J, qualified coverage, mean tardiness and final passing rate has its own
axis. Methods may be compared on the same metric. Provisional/complete statistics
report actual/planned coverage. Repetitions, worlds and training seeds must not
be flattened into independent observations. Rule does not acquire fictitious
training seeds merely by serving as a comparator.

Mobile layout is owned by W&B. Native metrics and custom web views have distinct
acceptance evidence; full mobile support for arbitrary custom panels is not
assumed. Entity diagnostics default to complete tables and selectable curves,
not hundreds of individually expanded panels.

## Evaluation identities

Limited evaluation remains the default. Completion evaluation is opt-in and may
continue until all orders reach Output, bounded by a protection horizon. The
default protection horizon is last arrival plus five frozen drain durations.
The common production window defaults to zero through last arrival plus one
frozen drain duration and never grows separately for a slower method.

Completion, protection truncation and wall-clock interruption are distinct.
Incomplete complete-cost J remains missing; accrued cost is a separate diagnostic.
Frozen inventories, executed scenarios and receipts must identify the same
realization and protocol. Future faults and arrivals remain hidden from policies.
Early confirmed drainage allows constant delivery counts beyond actual termination,
not fabricated resource trajectories; incomplete observations cannot be extended.

## Four research measures

Robustness compares each method to itself under paired normal and perturbed
conditions using a common physical window. A zero nominal output has no defined
ratio. Equal output at the window end is not evidence of equal completion times.

Adaptability uses separate controlled single-fault cases. Normal-load and
capacity-constrained supply regimes remain separate. Initial orders plus continuing
arrivals cover warm-up, baseline, disturbance and recovery. Storage capacity or a
multiple of it does not prove saturated throughput. Explicit candidate scans
produce calibration evidence; adopted settings are frozen, never silently tuned
during training or changed per method to obtain a convenient fault time.

A recovery protocol specifies baseline windows, minimum output, variation/trend
limits and observation bounds. There are no universal research thresholds for
these quantities. After an observed drop below 95% of the valid pre-fault baseline,
three nonoverlapping windows must again reach 95% to confirm recovery. Record both
first qualifying window end and confirmation. Primary elapsed time starts at
fault onset; repair-relative recovery is auxiliary. Adaptability is the reciprocal
of the first qualifying elapsed time, after confirmation. No observed degradation,
invalid baseline, insufficient supply, unrecovered and interrupted cases remain
distinct; none may be silently converted to success or removed from coverage.

SIG is positive for cost improvement: `(J_Base - J_SL) / J_Base`, alongside the
absolute `J_SL - J_Base`. Within each paired training seed, aggregate complete
matched case costs before division. Missing complete costs or zero Base cost do
not produce a relative score.

CAR is an operational common-target attainment ratio, not mathematical convergence.
The default target is a frozen Rule validation complete cost (configurable ratio),
never chosen from final test outcomes. Three validation nodes must qualify;
first attainment and confirmation budgets are distinct. The ratio uses confirmed
first attainment episodes. Unattained or zero-budget ratios remain explicitly
undefined. Sample/argmax and different validation panels are not mixed.

Task/team complexity, volatility, effective joint-action entropy and organizational
norm stability are outside this implementation. Existing counts or PPO policy
entropy must not be relabeled as those proposal measures.

## Verification and history

Pure metric tests require independent expected values, including missing data,
pairing, unequal repetitions, denominator zero and censored recovery. Native
integration verifies frozen receipts, continuation, monitoring RNG isolation,
completion protection and controlled recovery. Learning diagnostics reuse existing
forward outputs and leave losses, updates and random sequences unchanged.

Final acceptance requires independent engineering, science and testing reviews,
non-implementer cross-validation, and CI for the final commit. Temporary probes,
models and generated evidence stay outside the PR. Historical evidence remains
readable with explicit missing fields and its original source identity; this
decision does not reinterpret prior results or authorize legacy continuation.
