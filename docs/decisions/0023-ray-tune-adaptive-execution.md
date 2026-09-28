# 0023: Frozen experiment batches with measured adaptive execution

Status: accepted for the authorized single-node implementation.

Extends the composable learning and process-execution contracts in 0020–0022.
Supersedes the strict execution-identity requirement of 0013 **only** for new
`smartsom.adaptive-continuation/v1` runs. Legacy strict resume and already running
studies retain their original identity rules.

The user chooses scientific recipes before calibration. Ray Tune is optional
experiment orchestration, never simulator truth or algorithm-framework ownership
inside the core. Class Trainable exposes one complete native update, including
validation, best/patience changes and a complete checkpoint. FIFO plus a
version-locked ResourceChangingScheduler controls execution resources. Fresh
Tune segments restore native commits; Tuner.restore is not used to reinterpret
paused or historical scientific experiments.

The adaptive whitelist is numerical_threads and batch max_concurrent. All other
frozen inputs, algorithms, network/encoder contracts, seeds, budgets, sampling
streams/order and device cohort retain their identity. Numerical execution may
change floating-point rounding, so each acknowledged allocation is recorded;
thread changes are not a claim of bitwise-identical research outcomes.

Preflight checks the entire batch before allocating training. A prepared study
imports frozen inputs into a new run with new source provenance, and never
changes the original plan or resumes its old checkpoints. Calibration runs real
isolated workloads, retains engineering evidence and measures total useful
throughput under CPU/RAM/GPU peak constraints. Every group needs a valid baseline;
leaders need repeated measurements. Active calibration is bounded at 600 seconds;
resource waiting is cancellable and has no automatic deadline.

One driver owns the batch ledger, resource broker and presentation. Actors own
only their attempt's native records and checkpoints. Pending, staged, running
and stopping leases consume admission budgets until authoritative actor death
and owned-child exit are confirmed. Logical Ray CPU requests are paired with
actual Torch/BLAS thread limits; resource requests alone are not enforcement.
Office responsiveness uses resource reserve proxies, without promising desktop
frame rate. Resource changes never kill other applications or studies.

Atomic committed state carries source/scientific identity and checksums plus
complete learner, optimizer, sampler, RNG, replay, target-clock, best-selection
and patience state. Initial and best checkpoints, frozen partners and already
completed evaluation/control records travel with a portable continuation.
Successful experiment identities are never rerun on resume. A failure during
final evaluation can continue from training_complete without retraining.

Initial boundaries: local single node, CPU and explicitly allocated CUDA cohorts
run sequentially; one complete GPU per trial, no fractional GPU or MPS. Actual
parallel sampling, multi-node scheduling, scientific hyperparameter search,
GUI-latency guarantees and verified multi-GPU/H20/CARC performance are not part of
this acceptance. Their future changes require measured evidence and explicit
contracts rather than guessed hardware settings.
