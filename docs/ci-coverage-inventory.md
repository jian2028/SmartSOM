# CI coverage and ownership inventory

Baseline: integrated `main` commit `05af06c`. The maintained CI targets are **Ubuntu Linux and macOS only**, following the user's revised PR16 scope. Native Windows is excluded from the maintained platform union, not silently represented as covered. Windows hardware should run future experiments inside WSL2 Ubuntu with Linux Python; see [CI rules](ci.md).

All 173 pre-existing test files and all production `src/` files are preserved byte-for-byte against the baseline. Earlier PR16 Windows-only lifecycle repair and its new regressions were removed when the user withdrew native Windows maintenance. Existing Windows compatibility implementation and tests remain in the repository; they are not a native Windows support claim. The two new test files are CI selection regressions and the genuine public batch CLI acceptance, bringing primary ownership to 175 files.

## Before and after

- Old quality, learning and CP each broadly selected 173 files, or 519 file selections. One explicit primary partition replaces those broad overlaps, removing 346 repeated baseline file selections before intentional platform/minimal-environment repeats. This is selection arithmetic, not measured runtime speedup.
- Seven primary owners cover every test file once on Linux. Whole-file native ownership remains conservative: original 16,384/32,768-step acceptance budgets are unchanged, and affected learning can remain long. No 20-minute promise or full-only heavy partition is claimed.
- Supported platform repeats are explicit. Linux control (140 baseline nodes) and diagnostics (73) retain small overlaps for check continuity; macOS executes native control, diagnostics, checkpoint/background and fresh Optuna completion/resume contracts.
- Existing Windows-only cases remain collected by their Linux/macOS owners and may skip only under the exact existing OS reason allowances below. Native Windows runners, owners, force/console/publication qualification and new Windows maintenance are outside this PR's final support scope.

| Suite | Primary files | Maintained platforms | Environment / purpose |
| --- | ---: | --- | --- |
| fast | 80 | ubuntu-latest | Base only, lint and lean contracts |
| gym | 2 | ubuntu-latest | Minimal Gym-only adapter contracts |
| learning | 30 | ubuntu-latest | Real backend training, restore and process lifecycle |
| tuning | 22 | ubuntu-latest | Native continuation, batch CLI and both online Ray lifecycles |
| resource-learning | 7 | ubuntu-latest | Real resource PPO and protocol equivalence |
| usability | 26 | ubuntu-latest | Public lifecycle, search, reports and offline tracking |
| studio | 8 | ubuntu-latest | Installed offscreen Qt plus mixed native workflows |
| control | Intentional repeat | ubuntu-latest, macos-latest | Filesystem, process/control and display contracts |
| diagnostics | Intentional repeat | ubuntu-latest, macos-latest | Diagnostic/provenance contracts |
| macos-learning | Intentional repeat | macos-latest | Checkpoint Unicode/deep-path, background and Optuna completion/resume |

## Coverage and dependency boundaries

Learning/native owners install locked CPU learning, tuning, TensorBoard and Optuna dependencies. Resource-learning adds RLlib/PettingZoo; usability adds Matplotlib and offline W&B and requires Node for its DOM harness. Studio installs PySide6 and Ubuntu Qt system libraries and preflights concrete QtGui/QtWidgets imports. Offscreen acceptance is not real-window interaction or GPU/performance qualification.

The previously excluded Optuna parent-JSON test executes on Linux with search installed. Checkpoint Unicode/GBK/deep-path save and restore execute on Linux/macOS. CP remains the actual unsupported-grid negative-entrypoint contract, with incidental Matplotlib composition coverage explicitly assigned to usability; no new native CP solver acceptance is claimed. Gym retains its separate minimal environment. Both controlled-clock and real-clock native online Ray lifecycle cases have one primary tuning owner and preserve actor/state/cleanup evidence.

Every collected test must pass or match an exact supported-platform skip allowance. Missing packages, skipped modules, empty selection/execution, deselection, missing results and failed/cancelled required slices fail the aggregate. Baseline OS exceptions on Ubuntu/macOS are:

- `test_portable_control.py::test_windows_console_keys_and_interrupt`: `Windows console API`
- `test_portable_control.py::test_windows_force_does_not_kill_mismatched_birth`: `Windows handle identity regression`
- `test_probe_publication.py::test_native_windows_reader_release_allows_publication`: `native Windows file sharing`
- `test_probe_publication.py::test_native_windows_held_reader_fails_within_remaining_deadline`: `native Windows file sharing`

These exceptions describe unexecuted native Windows contracts, not Windows coverage. There are no maintained Windows-platform skip allowances. Unknown discovery/impact expands or fails closed as documented in the selector rules.

## Complete file mapping

The previous column is historical selection, not proof of execution. Its old Windows job names remain historical evidence only. The final platform-repeat column describes maintained Linux/macOS CI.

| Test file | Previous selected jobs | New primary | Platform repeats |
| --- | --- | --- | --- |
| `tests/integration/test_author_batch_cli_learning.py` | New coverage | tuning | None |
| `tests/integration/test_author_background.py` | quality, learning, cp | learning | macos-learning |
| `tests/integration/test_author_driver.py` | quality, learning, cp | fast | None |
| `tests/integration/test_author_learning.py` | quality, learning, cp, windows-learning | learning | None |
| `tests/integration/test_author_tuning.py` | quality, learning, cp | tuning | None |
| `tests/integration/test_automatic_inspection_ui.py` | quality, learning, cp | studio | None |
| `tests/integration/test_composable_study_parallel.py` | quality, learning, cp | learning | None |
| `tests/integration/test_cp_sat.py` | quality, learning, cp | fast | None |
| `tests/integration/test_experiment_api.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_grid_model_acceptance.py` | quality, learning, cp | learning | None |
| `tests/integration/test_historical_playback.py` | quality, learning, cp | studio | None |
| `tests/integration/test_learning_backends.py` | quality, learning, cp | learning | None |
| `tests/integration/test_learning_study_integration.py` | quality, learning, cp, usability | usability | macos-learning (fresh Optuna completion/resume node) |
| `tests/integration/test_production_examples.py` | quality, learning, cp | learning | None |
| `tests/integration/test_production_playback.py` | quality, learning, cp | studio | None |
| `tests/integration/test_production_workflows.py` | quality, learning, cp | studio | None |
| `tests/integration/test_resource_backend.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/integration/test_runtime_display_integration.py` | quality, learning, cp, windows-learning | usability | None |
| `tests/integration/test_studio.py` | quality, learning, cp | studio | None |
| `tests/integration/test_studio_editor.py` | quality, learning, cp | studio | None |
| `tests/integration/test_studio_symbols.py` | quality, learning, cp | studio | None |
| `tests/integration/test_task_performance_ui.py` | quality, learning, cp | studio | None |
| `tests/integration/test_training_extensions.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_training_lifecycle.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_training_probe.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_training_reward_isolation.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_training_sampling.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_tuning_batch_runtime.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_failure_exit.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_lazy_sampler_ownership.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_native_continuation.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_probe_native.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_ray_adapter.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_ray_native_resize.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_tuning_ray_online_lifecycle.py` | quality, learning, cp, tuning | tuning | None |
| `tests/integration/test_unified_stop.py` | quality, learning, cp | learning | None |
| `tests/test_package_import.py` | quality, learning, cp | fast | None |
| `tests/unit/test_api_workflow.py` | quality, learning, cp | fast | None |
| `tests/unit/test_arrival_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_arrivals.py` | quality, learning, cp | fast | None |
| `tests/unit/test_authoring.py` | quality, learning, cp | fast | None |
| `tests/unit/test_background.py` | quality, learning, cp | learning | macos-learning |
| `tests/unit/test_batch2_auto.py` | quality, learning, cp | fast | None |
| `tests/unit/test_batch_dashboard.py` | quality, learning, cp | fast | None |
| `tests/unit/test_buffer_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_buffer_lifecycle.py` | quality, learning, cp | fast | None |
| `tests/unit/test_buffers.py` | quality, learning, cp | fast | None |
| `tests/unit/test_catalog.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_checkpoint_portability.py` | quality, learning, cp | learning | macos-learning |
| `tests/unit/test_checkpoint_references.py` | quality, learning, cp | fast | None |
| `tests/unit/test_ci_selection.py` | New selector regression | fast | None |
| `tests/unit/test_compact_replay.py` | quality, learning, cp | learning | None |
| `tests/unit/test_composable_config.py` | quality, learning, cp | fast | None |
| `tests/unit/test_composable_learning.py` | quality, learning, cp, windows-learning | learning | None |
| `tests/unit/test_composable_physics.py` | quality, learning, cp | fast | None |
| `tests/unit/test_composable_workflow.py` | quality, learning, cp | learning | None |
| `tests/unit/test_composition_results.py` | quality, learning, cp | usability | None |
| `tests/unit/test_composition_tensorboard.py` | quality, learning, cp | fast | None |
| `tests/unit/test_console_resilience.py` | quality, learning, cp | fast | None |
| `tests/unit/test_diagnostics_config.py` | quality, learning, cp, diagnostics | learning | diagnostics |
| `tests/unit/test_dispatch_intentions.py` | quality, learning, cp | fast | None |
| `tests/unit/test_dispatcher_targets.py` | quality, learning, cp | learning | None |
| `tests/unit/test_doctor.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_domain.py` | quality, learning, cp | fast | None |
| `tests/unit/test_evaluation.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_experiment_boundaries.py` | quality, learning, cp | fast | None |
| `tests/unit/test_experiment_config.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_experiments.py` | quality, learning, cp | fast | None |
| `tests/unit/test_extended_diagnostics.py` | quality, learning, cp, diagnostics | learning | diagnostics |
| `tests/unit/test_extension_backends.py` | quality, learning, cp | usability | None |
| `tests/unit/test_extension_examples.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_extension_replay.py` | quality, learning, cp | learning | None |
| `tests/unit/test_extensions.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_factory_design.py` | quality, learning, cp | fast | None |
| `tests/unit/test_factory_design_config.py` | quality, learning, cp | fast | None |
| `tests/unit/test_factory_reliability.py` | quality, learning, cp | fast | None |
| `tests/unit/test_fjsp_engine.py` | quality, learning, cp | fast | None |
| `tests/unit/test_fjsp_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_fjsp_solvers.py` | quality, learning, cp | fast | None |
| `tests/unit/test_four_config.py` | quality, learning, cp | fast | None |
| `tests/unit/test_grid_composition.py` | quality, learning, cp | fast | None |
| `tests/unit/test_grid_timeline.py` | quality, learning, cp | fast | None |
| `tests/unit/test_holding.py` | quality, learning, cp | fast | None |
| `tests/unit/test_holding_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_idetc_acceptance.py` | quality, learning, cp | fast | None |
| `tests/unit/test_job_content.py` | quality, learning, cp | learning | None |
| `tests/unit/test_learning_cli.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_learning_gymnasium.py` | quality, learning, cp, gym | gym | None |
| `tests/unit/test_learning_projection.py` | quality, learning, cp, gym | gym | None |
| `tests/unit/test_learning_study.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_machine_event_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_machine_events.py` | quality, learning, cp | fast | None |
| `tests/unit/test_online_concurrency.py` | quality, learning, cp | fast | None |
| `tests/unit/test_operation_catalog.py` | quality, learning, cp | fast | None |
| `tests/unit/test_optuna_search.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_packaging.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_performance_profiles.py` | quality, learning, cp | fast | None |
| `tests/unit/test_performance_rows.py` | quality, learning, cp | fast | None |
| `tests/unit/test_physical_job_observation.py` | quality, learning, cp, diagnostics | learning | diagnostics |
| `tests/unit/test_port_owner_index.py` | quality, learning, cp | fast | None |
| `tests/unit/test_port_target_decode.py` | quality, learning, cp | fast | None |
| `tests/unit/test_portable_control.py` | quality, learning, cp, control | fast | control |
| `tests/unit/test_portable_paths.py` | quality, learning, cp, control | fast | control |
| `tests/unit/test_preflight_timeline.py` | quality, learning, cp | fast | None |
| `tests/unit/test_probe_publication.py` | quality, learning, cp | tuning | control |
| `tests/unit/test_processing_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_processing_times.py` | quality, learning, cp | fast | None |
| `tests/unit/test_production_config.py` | quality, learning, cp | fast | None |
| `tests/unit/test_production_diagnostics.py` | quality, learning, cp, diagnostics | learning | diagnostics |
| `tests/unit/test_production_learning.py` | quality, learning, cp | learning | None |
| `tests/unit/test_production_runtime.py` | quality, learning, cp | fast | None |
| `tests/unit/test_quality.py` | quality, learning, cp | fast | None |
| `tests/unit/test_quality_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_quickstart.py` | quality, learning, cp | fast | None |
| `tests/unit/test_replay_clock.py` | quality, learning, cp | fast | None |
| `tests/unit/test_replay_evidence.py` | quality, learning, cp | fast | None |
| `tests/unit/test_replay_model.py` | quality, learning, cp | fast | None |
| `tests/unit/test_report.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_report_evaluation.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_resource_acceptance.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/unit/test_resource_lifecycle.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/unit/test_resource_optimization.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/unit/test_resource_projection.py` | quality, learning, cp, resource-protocol, resource-learning | resource-learning | None |
| `tests/unit/test_resource_protocol.py` | quality, learning, cp, resource-protocol, resource-learning | resource-learning | None |
| `tests/unit/test_resource_training.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/unit/test_rule_registry.py` | quality, learning, cp | fast | None |
| `tests/unit/test_run_control.py` | quality, learning, cp, control | fast | control |
| `tests/unit/test_runtime_display.py` | quality, learning, cp, control | fast | control |
| `tests/unit/test_runtime_optimization_equivalence.py` | quality, learning, cp | learning | None |
| `tests/unit/test_sampling_failures.py` | quality, learning, cp | learning | None |
| `tests/unit/test_sb3_metrics.py` | quality, learning, cp | learning | None |
| `tests/unit/test_schedule_replay.py` | quality, learning, cp | fast | None |
| `tests/unit/test_small_study.py` | quality, learning, cp | learning | None |
| `tests/unit/test_solvers.py` | quality, learning, cp | fast | None |
| `tests/unit/test_static_engine.py` | quality, learning, cp | fast | None |
| `tests/unit/test_structural_snapshot.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studies.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studio_cli.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studio_editing.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studio_template.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studio_template4.py` | quality, learning, cp | fast | None |
| `tests/unit/test_studio_template789.py` | quality, learning, cp | fast | None |
| `tests/unit/test_study_execution.py` | quality, learning, cp | fast | None |
| `tests/unit/test_study_progress.py` | quality, learning, cp | fast | None |
| `tests/unit/test_task_performance.py` | quality, learning, cp | fast | None |
| `tests/unit/test_template1_demo.py` | quality, learning, cp | fast | None |
| `tests/unit/test_trace_cursor.py` | quality, learning, cp | fast | None |
| `tests/unit/test_training.py` | quality, learning, cp | learning | None |
| `tests/unit/test_training_controls.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_training_display.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_transport.py` | quality, learning, cp | fast | None |
| `tests/unit/test_transport_inputs.py` | quality, learning, cp | fast | None |
| `tests/unit/test_travel_time_matrix.py` | quality, learning, cp | fast | None |
| `tests/unit/test_tuning_batch.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_broker.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_calibration.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_callbacks.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_dashboard.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_online_context.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_probe.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_ray.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_resources.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_session.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_tuning_session_ownership.py` | quality, learning, cp, tuning | tuning | None |
| `tests/unit/test_unified_commands.py` | quality, learning, cp | fast | None |
| `tests/unit/test_usability_acceptance.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_v3_blind_output.py` | quality, learning, cp | learning | None |
| `tests/unit/test_v3_diagnostic_contracts.py` | quality, learning, cp, diagnostics | learning | diagnostics |
| `tests/unit/test_v3_dispatch_liveness.py` | quality, learning, cp | fast | None |
| `tests/unit/test_v3_model_compatibility.py` | quality, learning, cp | learning | None |
| `tests/unit/test_v3_quality_routing.py` | quality, learning, cp | fast | None |
| `tests/unit/test_v3_task_reward.py` | quality, learning, cp | learning | None |
| `tests/unit/test_validation_cases.py` | quality, learning, cp, usability | usability | None |
| `tests/unit/test_workflow_dashboard.py` | quality, learning, cp, control | learning | control |
| `tests/unit/test_workload_v3.py` | quality, learning, cp | fast | None |

## Public directory-batch acceptance

`test_author_batch_cli_learning.py` uses actual public subprocess `batch-run`, `stop`, and fresh-process `resume` commands with one real Ray-queued SB3 learner. Its engineering fixture has one thread/environment, no sampler subprocesses, calibration off, and 512 ticks/64 updates. It requires a verified nonterminal commit, advancing resume, exact native model/optimizer/RNG/simulator state including tensor dtype/shape, and final held-out evaluation case/seed identity with no engineering failure. Numerical equality of evaluation metrics to the reference is not claimed.

The public stop is cooperative, not an exact-update hook. It waits for committed progress and fails if training completes before partial stop; there is no retry or checkpoint rewind to conceal a race. Subprocess waits/reference updates are bounded and emergency cleanup targets observed child PID/birth identities. This case passed on GitHub at `90af088` in 82.04 seconds; final-head qualification is reported separately.

## Verification scope and migration

Fully provisioned primary-union collection must match whole-repository collection on Linux, with no missing, duplicate or unexpected nodes. This establishes selector completeness for maintained owners, not native Windows execution. The final suite must run on its exact PR head; older run outcomes do not satisfy new required checks.

The removed check contexts are `windows-learning`, `control (windows-latest)` and `Native diagnostics and provenance (windows-latest)`. No branch-protection or repository-rule setting is changed. If any removed context is required, merging will need a separately approved required-check migration. Existing Windows runs/experiments and implementation are not rewritten.
