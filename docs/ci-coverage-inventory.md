# CI coverage and ownership inventory

Baseline: integrated `main` commit `05af06c`. This is a static ownership and dependency inventory, not a claim that the new native/platform jobs have already passed. The older CI audit reviewed 173 test files; this checkout also has 173 existing test files. The new selector regression file brings the declared primary inventory to 174. Supporting Python modules without test-file names are imported by their owning tests and are not standalone pytest selectors.

## Before and after

- Previously quality, learning and CP each collected all 173 files. Their executed nodeids differed with installed extras; collecting a file did not prove its optional contracts ran. Tuning repeated the 19 tuning files, resource protocol repeated two resource-learning files, and usability repeated 23 files with a different dependency set.
- Now seven explicit primary suite lists partition every test file exactly once. Mixed files remain conservatively native. There is no `pytest .`, test-directory blanket selector, or `-m "not learning"` shortcut. Cross-platform repetition is declared separately and intentionally.
- Both controlled-clock and real-clock online Ray lifecycle tests remain together in the tuning owner on native-impact and full runs. No other suite selects that file. Existing PPO/DQN continuation, optimizer/RNG/replay/ledger/validation assertions remain unchanged.
- The `gym` suite retains its minimal Gym-only environment, separate from full learning. The fast suite installs no optional extras and retains package-import/four-file blocked-import boundaries.
- CP is the existing unsupported-grid negative-entrypoint contract, retained in fast with the fake-backend FJSP unit contracts. No native CP solver acceptance is claimed. The formerly incidental Matplotlib composition-results test now belongs to usability with the reports extra before removing the broad CP environment.

| Suite | Primary files | Platforms | Environment / purpose |
| --- | ---: | --- | --- |
| fast | 80 | ubuntu-latest | Base only; bounded base contracts |
| gym | 2 | ubuntu-latest | gym; minimal Gym adapter contracts |
| learning | 30 | ubuntu-latest | cpu, learning, tuning, tensorboard, search; real CPU training, restore and process lifecycle |
| tuning | 21 | ubuntu-latest | cpu, learning, tuning, tensorboard, search; real PPO/DQN continuation, actors and both bounded online lifecycles |
| resource-learning | 7 | ubuntu-latest | cpu, learning-marl, tensorboard; real resource PPO lifecycle and protocol equivalence |
| usability | 26 | ubuntu-latest | cpu, learning, tuning, tensorboard, search, reports, wandb; public native lifecycle, search, reporting and offline tracking |
| studio | 8 | ubuntu-latest | cpu, learning, tuning, tensorboard, search, studio; offscreen Qt rendering plus mixed real learning workflows |
| control | Intentional repeat | ubuntu-latest, windows-latest, macos-latest | cpu, learning, tuning, tensorboard, search; intentional native platform contract repetition |
| windows-learning | Intentional repeat | windows-latest | cpu, learning, tuning, tensorboard, search; intentional native platform contract repetition |
| diagnostics | Intentional repeat | ubuntu-latest, windows-latest, macos-latest | cpu, learning, tuning, tensorboard, search; intentional native platform contract repetition |
| macos-learning | Intentional repeat | macos-latest | cpu, learning, tuning, tensorboard, search; intentional native platform contract repetition |

Suite costs are qualitative. No duration reduction or timeout sizing is asserted without measured distributions. `full` means the union of these explicit primary and platform suites, not a second broad pytest run.

## Closed selection gaps

- Control preserves all five previous portable selectors and adds `test_probe_publication.py` on Linux, Windows and macOS. This retains real advisory locking, process birth/handle identity and native Windows sharing/publication deadlines. Its existing workflow-dashboard file includes real SB3 work, so the environment now supplies full native prerequisites rather than silently skipping those cases.
- Windows-learning preserves composable learning, author learning and runtime-display integration. The former `not fresh_optuna` exclusion is removed: search is installed and the Optuna parent JSON lifecycle executes.
- Checkpoint portability is owned by primary Linux learning and repeated on Windows-learning and macOS-learning. The whole file retains UTF-8 under GBK defaults, Unicode digests, deep path copies, genuine save/restore and deep Ray/PyArrow synchronization. These are actual host runs, not Linux-only path emulation.
- Diagnostics preserves all five existing files on all three operating systems.
- MacOS-learning adds the actual four-file author background integration and the real detached-child background contract, alongside checkpoint portability. Unsupported POSIX background execution is not assigned to Windows.
- Studio installs PySide6 and full learning for its eight integration files, including the mixed production-workflows file. Offscreen execution establishes Qt contracts only; it does not establish real-window interaction.

## Dependencies and skip enforcement

All declared native imports are prerequisites, including transitive NumPy, SciPy and PyArrow where exercised. Learning/tuning/platform suites provision learning, tuning, CPU Torch, TensorBoard and Optuna; usability additionally provisions Matplotlib and offline W&B. Studio additionally provisions PySide6. Resource-learning supplies its real RLlib/PettingZoo/TensorBoard environment. The reports DOM harness also requires the Node executable; usability must check it before running.

Every collected skip is a failure unless its exact expected reason, nodeid prefix and actual OS match the manifest. Missing packages, skipped modules and missing Node are never permitted. The only allowed reasons are:

- `tests/unit/test_portable_control.py::test_windows_console_keys_and_interrupt`: `Windows console API` on ubuntu-latest, macos-latest
- `tests/unit/test_portable_control.py::test_windows_force_does_not_kill_mismatched_birth`: `Windows handle identity regression` on ubuntu-latest, macos-latest
- `tests/unit/test_probe_publication.py::test_native_windows_reader_release_allows_publication`: `native Windows file sharing` on ubuntu-latest, macos-latest
- `tests/unit/test_probe_publication.py::test_native_windows_held_reader_fails_within_remaining_deadline`: `native Windows file sharing` on ubuntu-latest, macos-latest
- `tests/unit/test_portable_paths.py::test_read_model_rejects_invalid_parent_before_dotdot`: `POSIX component traversal semantics` on windows-latest
- `tests/unit/test_portable_paths.py::test_read_model_wraps_symlink_loop`: `POSIX symlinks require no privilege` on windows-latest

The Windows symbolic-link privilege skips in authoring/tuning-session are not allowlisted: those files are currently primary Linux selectors only. POSIX-only author-background and detached-child tests run only on Linux/macOS. An OS/dependency expansion must review those reasons explicitly rather than permitting arbitrary skips.

## Complete file mapping

The old column records selected files, not proven execution. All old rows include the broad quality/learning/CP collection. Portable jobs repeat per their OS matrices. The new column is the sole primary owner; platform repeats are listed separately.

| Test file | Previous selected jobs | New primary | Platform repeats |
| --- | --- | --- | --- |
| `tests/integration/test_author_background.py` | quality, learning, cp | learning | macos-learning |
| `tests/integration/test_author_driver.py` | quality, learning, cp | fast | None |
| `tests/integration/test_author_learning.py` | quality, learning, cp, windows-learning | learning | windows-learning |
| `tests/integration/test_author_tuning.py` | quality, learning, cp | tuning | None |
| `tests/integration/test_automatic_inspection_ui.py` | quality, learning, cp | studio | None |
| `tests/integration/test_composable_study_parallel.py` | quality, learning, cp | learning | None |
| `tests/integration/test_cp_sat.py` | quality, learning, cp | fast | None |
| `tests/integration/test_experiment_api.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_grid_model_acceptance.py` | quality, learning, cp | learning | None |
| `tests/integration/test_historical_playback.py` | quality, learning, cp | studio | None |
| `tests/integration/test_learning_backends.py` | quality, learning, cp | learning | None |
| `tests/integration/test_learning_study_integration.py` | quality, learning, cp, usability | usability | None |
| `tests/integration/test_production_examples.py` | quality, learning, cp | learning | None |
| `tests/integration/test_production_playback.py` | quality, learning, cp | studio | None |
| `tests/integration/test_production_workflows.py` | quality, learning, cp | studio | None |
| `tests/integration/test_resource_backend.py` | quality, learning, cp, resource-learning | resource-learning | None |
| `tests/integration/test_runtime_display_integration.py` | quality, learning, cp, windows-learning | usability | windows-learning |
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
| `tests/unit/test_checkpoint_portability.py` | quality, learning, cp | learning | windows-learning, macos-learning |
| `tests/unit/test_checkpoint_references.py` | quality, learning, cp | fast | None |
| `tests/unit/test_ci_selection.py` | New selector regression | fast | None |
| `tests/unit/test_compact_replay.py` | quality, learning, cp | learning | None |
| `tests/unit/test_composable_config.py` | quality, learning, cp | fast | None |
| `tests/unit/test_composable_learning.py` | quality, learning, cp, windows-learning | learning | windows-learning |
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

## Review and acceptance boundary

Ownership is guarded by the selector regression tests and manifest validation: a new or renamed test file must acquire one primary owner, selectors must remain explicit, and invalid dependency/platform data must fail closed. The initial inventory intentionally uses file-level ownership so future tests added to an existing file inherit its provisioned suite. Runtime JUnit evidence, strict skip validation and native/platform CI runs are still necessary before calling this execution coverage verified. No simulator source, existing test assertion, numeric tolerance or research milestone changes are part of this inventory.

## Local collection verification

On 2026-10-11, a separate Python 3.12 environment was provisioned from this
checkout with `uv sync --locked` and the CPU, learning, tuning, TensorBoard,
search, reports, W&B and Studio extras. The base environment remained lean.
Whole-repository and seven separate primary-selector `--collect-only` passes
produced these nodeid counts:

| Collection | Nodeids |
| --- | ---: |
| Whole repository | 3,279 |
| fast | 1,758 |
| gym | 32 |
| learning | 449 |
| tuning | 376 |
| resource-learning | 106 |
| usability | 392 |
| studio | 166 |

The primary nodeid union exactly equalled the whole-repository collection:
zero missing, unexpected or duplicate nodeids and zero collection skips or
errors. This checks expanded parametrized identities rather than only static
file counts. It establishes selection completeness in the fully provisioned
Linux environment, not passing test execution, platform behavior or runtime
speed. Native actor/training execution and hosted Windows/macOS verification
remain separate acceptance steps.

The final collection includes 3,223 nodeids from the 173 baseline test files
and 56 new CI-selector regression cases. Each of the 173 existing test files
was compared with its Git blob at `05af06c`; all hashes match exactly.
Existing tracked `tests/` files and all `src/` files are unchanged from that
baseline; the only added test file is the CI regression file. No existing
assertion, fixture or production implementation was removed or weakened. These unchanged baseline files account for every collected nodeid
other than the new CI regression file. This is byte-preservation evidence plus
current expanded collection, not a claim of a second historical runtime run.

## Intentional remaining Linux overlap

The old three broad jobs selected each of the 173 baseline files three times
(519 file selections). The new primary partition selects each baseline file
once, eliminating 346 repeated broad file selections before platform and
minimal-dependency checks. This is a selector comparison, not a runtime speedup.

For continuity of established platform check contexts, Linux still repeats the
explicit control and diagnostics slices: `control` has 6 files / 140 collected nodes, `diagnostics` has 5 files / 73 collected nodes.
Some control tests add evidence in a native dependency environment versus fast;
the native diagnostics overlap is deliberate continuity, not unique additional
coverage. Windows/macOS repetitions remain required platform evidence. The
online lifecycle tests are not part of these repeats and execute only in tuning.
