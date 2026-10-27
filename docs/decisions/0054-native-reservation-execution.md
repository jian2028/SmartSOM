# 0054 — Native reservation execution and shared performance paths

Date: 2026-10-25
Status: User-authorized implementation; development verification only.

For the reservation contract, supersedes ADR0053's separate-entrypoint restriction.
The public `smartsom check/run` and `check_experiment/run_experiment` APIs dispatch
explicit `smartsom.base-experiment/v1` inputs to the reservation protocol. Existing
v4 inputs retain their own semantics. This does not reinterpret old intention
policies or migrate checkpoints between action contracts.

Factory remains the shared `smartsom.factory/v2` design. Workload, Algorithm and
Experiment own their existing typed fields; Experiment's `run` selects smoke,
calibration, training or evaluation. The compiler detaches the four documents,
validates overrides and geometry, and records source hashes. Execution uses the
same ProductionSimulator, BaseProtocol, semantic candidates, SharedMover and
BasePPO. `reservation_runtime` owns the common rollout and learning lifecycle;
the old diagnostic module is a compatibility entrypoint, not another simulator.

Native execution is foreground and uses existing cooperative stop ownership.
Automatic resume, background launch, v4 matrix/batch/Tune and Studio policy
editing are not supported for this contract yet; unsupported CLI options fail
explicitly. These limitations must not be advertised as completed integration.
Training/evaluation require a successful scenario-matching calibration. Bounded
smoke requires no calibrated drain claim and records budget cuts as incomplete.
No new scientific algorithm, reward, STOP restriction or Mover is introduced.

The base is integrated on the optimized main source. Actor-only evaluation omits
critic feature construction and inference. PPO caches only immutable raw input
tensors within one group/update, never parameter-dependent embeddings or mutable
same-tick protocol state. State-dict keys, candidate identities, RNG order,
optimizer ordering and masks are retained and checked against the uncached path.
Measured performance belongs to its fixed workload and source; merged optimization
history alone does not imply those paths execute or accelerate this protocol.
