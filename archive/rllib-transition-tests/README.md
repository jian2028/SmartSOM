# RLlib transition test provenance

These files preserve the pre-core-closeout test definitions. They are historical
snapshots, outside current pytest/CI selectors; they are not skipped current
acceptance tests. The full legacy runtime is retained separately under
`archive/legacy_runtime`. Current simulator and observation unit contracts remain
active. ADR0098 records the user-approved scope.

| Historical contract | Current disposition / evidence |
| --- | --- |
| `test_reservation_lifecycle`: calibration extension/cap and malformed limit | Kept active; scientific rule calibration is unchanged. |
| Same file: case receipts and completed calibration restore | `test_rllib_factory_training::test_calibration_tune_checkpoint_restores_receipts_without_repeating_cases` verifies official checkpoint recovery after original receipts are removed, remaining cases only and completed idempotency. |
| Same file: exact native update/wave state, `last.pt`, generation pointers | Retired mechanism; official PPO/DQN update and replay recovery in `test_rllib_factory_training` and `test_rllib_dqn_checkpoint`. No numerical equivalence claim across frameworks. |
| Same file: public budget, background and whole-run control | Kept/migrated in the active file: budget then dependent stage; real CLI pause, fresh-process resume, stop, deadline/event identity and child release. |
| Same file: standalone evaluation, common window, CAR/recovery checkpoint scans | Deferred. Public preflight rejects these requests; retained here, not implemented via a native fallback. |
| `test_reservation_tune`: old actor phases and native retry inputs | Active file now tests three official trials at concurrency two, completed restore idempotency, JSON output, PPO finite retry success/exhaustion and DQN explicit retry with learner/replay comparisons. |
| `test_tune_phase_execution`: phase rebuilds/native exact episode checkpoints | Retired. Resident official components tested by the active cohort and RLlib factory tests. |
| Same file: search resume | Active file uses real CLI processes and three-objective Optuna, frozen calibration provenance, completed candidate preservation and tracking deduplication. Censor/Pareto tests retained. Budget-before-panel coverage uses the existing `test_public_cli_runs_official_rllib_trial[not_evaluated]`. |
| Same file: native actor cleanup/corrupt generation restore | Retired adapter. Official checkpoint corruption/source rejection remains covered by RLlib factory/DQN checkpoint tests and public configuration tests. |
| `test_tune_recovery_evaluation` | All three cases exercise deferred standalone/window/recovery analysis. Removed from current ownership; rejection paths remain active. |
| `test_research_nodes`: exact physical nodes | Native exact-boundary behavior retired; official iteration overshoot is accepted. |
| Same file: public study models/panels/result guard | Current public calibration/train/final panel test, cohort test and public run-existing-output guard cover retained execution. Old native report layouts are not a new framework contract. |
| Same file: generated standalone Rule test panel | Deferred along with full comparator evaluation. |
| `test_current_runtime`: centralized rule calibration | Kept smoke/calibration and wrong-control provenance rejection. Standalone evaluation part deferred. Physics, fixed rules and historical trace-reader tests remain active. |
| Same file: native parallel standalone evaluation receipts | Deferred; official fixed panel concurrency has separate RLlib/cohort tests. |
| `test_controller_comparison`: public central+Justin smoke | Justin rule smoke kept; central learning migration deferred. Native checkpoint public resume is explicitly rejected without modifying artifacts. Historical algorithm and frozen-inventory checks use raw document compilation, not the current launch preflight; they do not establish RLlib acceptance. |
| `test_reservation_native`: native calibration/train/standalone evaluate chain | Replaced by the public scientific calibration → official PPO → periodic and final best/last chain. Existing native model unit checks are legacy regression coverage, not migration evidence. |

Passing counts from retained native-internal tests must not be reported as RLlib
acceptance. Each active CI selector still requires its declared dependencies;
no blanket skip or assertion relaxation substitutes for a required core scenario.


The dedicated `test_training_sampling_controls.py` snapshot preserves native
fractional update credit, target clocks, synchronous waves, exact wave rollback,
and new training recording. These retired mechanisms are excluded from both
`tuning` and `macos-learning`. Official update/replay recovery is covered by the
RLlib factory and DQN checkpoint tests; fragment trajectory integrity by
`test_fragment_admission_preserves_uncut_agent_trajectories`; public budgets,
errors and release by lifecycle/cohort/retry tests.

`test_tune_evaluation.py` preserves the former custom case-actor pool contract.
The current evaluator uses resident official EnvRunners; fixed panels, order,
missing/duplicate rejection and truncation selection are covered by
`test_rllib_evaluation` and real fixed-panel tests. This does not claim a forced
kill test of every official evaluation worker. Neither archived file supplies
helpers to current tests.
