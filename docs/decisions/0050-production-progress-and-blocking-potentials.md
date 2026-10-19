# 0050 — Training production progress and blocking potentials

Date: 2026-10-13
Status: User-authorized local implementation; integration and formal runs pending.

Extends ADR0039 and ADR0044 for explicitly selected training-only physical
potentials. It supersedes only the prohibition on stacking these new conversions
with the task. Legacy team/role transforms remain incompatible. The Base task,
quality smoothing, public policy inputs, simulator, action rules, evaluation and
best checkpoint metric retain their existing semantics. Engineering tests are
not evidence of better learning or multiagent convergence.

`learner.reward.production_potential_shaping` declares
`smartsom.production-potential-training/v1`, nonnegative progress/blocking
weights, nominal handoff units and public pressure age/occupancy thresholds.
It requires the explicit V3 shipment task and gamma1. V4 carries it through the
ordinary author compiler into TrainingSession, without a custom learner or core.

Progress is the sum over original released orders of normalized reference work
and required handoffs, divided by the task's fixed reference_jobs. Shipped
original orders contribute one each, irrespective of latent quality. Outside
FIFO contributes zero until entering visible physical state. Original public
reference_ticks (otherwise nominal_ticks) weight completed operations. Current
processing uses capped public active elapsed ticks against canonical reference
work; it uses neither remaining sampled duration nor a mode-dependent nominal.
For n operations, n+1 handoffs are required: delivery to each processing stage,
then Output. A PRE/READY/PROCESSING location credits the next processing delivery;
BLOCKED/POST/onboard/inspection retain only already completed stage handoffs.
Pickup, geometric motion, retarget and optional inspection loops earn no new
handoff. Only the current attempt participates; known FAIL contributes zero.
Replacement cannot accumulate progress across discarded attempts.

Blocking is minus the mean of three normalized pressures: BLOCKED machine age
averaged over all machines; finite PRE/POST occupancy times oldest resident age
above an occupancy threshold, averaged over finite PRE/POST slots; and cargo age
beyond a fixed grace, averaged over all AGVs. Ages are clipped at their scale.
BLOCKED clocks key machine and current job, clearing at physical unblock.
Cargo age uses the job's pickup timestamp, which retarget cannot reset. No
policy intentions, future arrivals, oracle quality or realized remaining work
enter either potential. Immutable factory roles/capacities only classify public
state; changing these calculations does not alter the physical protocol.

At physical interval boundaries, let raw potentials be P and B, and define
centered Phi_P(t)=P(t)-P(0), Phi_B(t)=B(t)-B(0). Emit
`r_train = r_base_train + wp*(Phi_P(t+1)-Phi_P(t))
                          + wb*(Phi_B(t+1)-Phi_B(t))`.
The genuine finite task terminal has both potentials zero, including time-limit
truncation. Its extra correction removes the raw centered terminal potentials.
Thus added rewards sum to zero over each complete gamma1 objective episode;
combined with ADR0044 the episode return equals original J exactly. A partial
training budget, update cut, validation, checkpoint or pause does not fabricate
terminal compensation. Shared role/group copies receive the same team interval;
accounting reports use one physical environment/episode ledger.

Contracts, partial potentials, BLOCKED clocks and underlying quality ledger are
saved in FULL continuations. Restore rejects changed contracts and inconsistent
totals before learner state loads, and validates active potentials against saved
public simulator state. Models and run records identify the selected training
conversion. Original task J remains the evaluation/checkpoint-selection metric.

The first Batch05 candidates use progress_weight=0.1, blocking_weight=0.1 when
enabled, handoff_reference_ticks=16, blocking_age_ticks=32,
buffer_pressure_threshold=0.75 and cargo_grace_ticks=64. These are explicit
uncalibrated initial candidates; they are neither tuned optima nor research
results. The paired study retains Base seeds 101/102/103, frozen worlds, timing,
initial packages, all 22 training groups, budgets and validation/evaluation.

Formal execution requires a clean integrated main revision or explicit tag.
Static transfer packages may be prepared earlier but must be checked and copied
out of transfer before execution. Existing active Base runs are not resumed
under this changed source. Three seeds per reward group are scientific inputs;
concurrency belongs to the native live queue introduced by ADR0047. No shaping
launcher chooses a limit or starts extra batch coordinators. The operator changes
the native queue limit at runtime; native guards remain authoritative.
