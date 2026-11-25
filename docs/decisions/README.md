# Decision records

Each record states one decision and its scope at the time it was written. A
record is history: later records amend it, and none is edited to match the
current code. `docs/architecture.md` and `docs/roadmap.md` describe the current
state.

## When to write one

Write a record when a change alters paper-facing semantics (reward,
observation, metrics, termination, comparison fairness) or a public contract
that existing inputs, runs or checkpoints depend on. Record other design
choices in the pull request description.

## Numbering

Give a new record the next number after the highest one on `main` at merge
time. If another record lands first, renumber on the branch before merging.
Add the row to the index below in the same change.

Four numbers are shared by two records each (0036, 0065, 0066, 0077). They
keep their file names because other records and run manifests cite them; cite
these by file name, not by number alone. Numbers 0030 and 0072 have no record
in the tracked tree.

## Index

"Status" is the record's own status line, quoted as written. "Later records"
lists records whose supersession statement names this one. Most replace,
narrow or extend only one portion, so read the later record before treating
the earlier one as obsolete.

| Record | Title | Status | Later records |
| --- | --- | --- | --- |
| [0001](0001-project-foundation.md) | Project Foundation | Accepted | — |
| [0002](0002-experiment-configuration-and-execution.md) | Experiment Configuration and Execution Contract | Accepted | [0004](0004-paired-ablation-inputs.md) |
| [0003](0003-online-arrival-timing-and-visibility.md) | Independent arrival timing and policy visibility | Accepted; implements the Week2 item 5 contract. | — |
| [0004](0004-paired-ablation-inputs.md) | Preserve unchanged inputs in paired ablations | Accepted design contract; study/batch execution is not implemented. | [0013](0013-experiment-usability-and-training-lifecycle.md) |
| [0005](0005-machine-outages-and-processing-progress.md) | Machine outages and processing progress | Accepted; implements the Week2 item 7 contract. | [0007](0007-finite-buffers-and-blocking.md) |
| [0006](0006-fixed-matrix-transport-and-execution-replay.md) | Fixed-matrix AGV transport and complete execution replay | Accepted for Week2 item 8. | [0017](0017-grid-production-and-playback.md) |
| [0007](0007-finite-buffers-and-blocking.md) | Finite buffers, blocking and exact execution replay | Accepted for Week2 item 9. | [0017](0017-grid-production-and-playback.md) |
| [0008](0008-quality-speed-and-final-inspection.md) | Quality-speed capabilities and final inspection | Accepted | — |
| [0009](0009-paired-studies-and-recovery.md) | Paired studies and restartable single-host execution | Accepted and implemented. | [0013](0013-experiment-usability-and-training-lifecycle.md) |
| [0010](0010-shared-holding-buffer.md) | Shared holding buffer for AGV congestion fallback | Accepted and implemented for the IDETC integration slice. | — |
| [0011](0011-centralized-learning-projection-and-episodes.md) | Centralized learning projection and episode contract | Accepted. Shared projection/Gym and optional framework training/checkpoint evaluation are separate implementation checkpoints with separate acceptance evidence. | [0013](0013-experiment-usability-and-training-lifecycle.md) |
| [0012](0012-resource-agents-and-joint-proposals.md) | Resource agents, public observations and joint proposals | Accepted. Projection/Parallel API and actual RLlib training are separate implementation checkpoints. This extends ADR 0011 without changing its centralized projection, the simulator, scientific seeds or physical replay contract. | [0020](0020-composable-policies-and-physical-tick-learning.md) |
| [0013](0013-experiment-usability-and-training-lifecycle.md) | Editable experiment recipes and a shared training lifecycle | Accepted design; implementation and platform qualification are staged. | [0020](0020-composable-policies-and-physical-tick-learning.md) |
| [0014](0014-studio-factory-design.md) | Studio factory design and read-only preview | Accepted | [0015](0015-studio-static-editor.md), [0017](0017-grid-production-and-playback.md), [0018](0018-multiple-system-buffers.md), [0062](0062-scenario-authoring-and-studio.md), [0066](0066-four-map-catalog.md) |
| [0015](0015-studio-static-editor.md) | Complete static editor and document lifecycle | Accepted | [0062](0062-scenario-authoring-and-studio.md), [0066](0066-four-map-catalog.md) |
| [0016](0016-factory-operation-catalog.md) | Factory operation catalog and portable authoring preferences | Accepted for the static Studio configuration slice. | [0017](0017-grid-production-and-playback.md) |
| [0017](0017-grid-production-and-playback.md) | One grid production contract with live and recorded views | Execution contract accepted and implemented. Development verification and remaining native interaction limits are listed in [implementation status](https://github.com/jian2028/SmartSOM/blob/5efd4d0bdde33c73a7528c3278a8307752a483cf/docs/production-runtime.md). Local commits do not promote historical research acceptance results. | [0019](0019-automatic-inspection-and-local-disposal.md), [0020](0020-composable-policies-and-physical-tick-learning.md), [0021](0021-travel-time-matrix-and-small-hv-study.md), [0038](0038-v3-blind-output-and-privileged-quality.md), [0058](0058-current-runtime-and-tracking.md) |
| [0018](0018-multiple-system-buffers.md) | Multiple system input and output facilities | Accepted for the requested larger Studio factory templates. | — |
| [0019](0019-automatic-inspection-and-local-disposal.md) | Automatic per-job inspection and local disposal | not stated | — |
| [0020](0020-composable-policies-and-physical-tick-learning.md) | Composable policies and physical-tick learning | Accepted implementation scope; verification is reported separately. | [0021](0021-travel-time-matrix-and-small-hv-study.md), [0025](0025-four-file-authoring-and-background.md), [0035](0035-v3-persistent-dispatch-intentions.md), [0037](0037-v3-service-and-learning-liveness.md), [0038](0038-v3-blind-output-and-privileged-quality.md), [0058](0058-current-runtime-and-tracking.md) |
| [0021](0021-travel-time-matrix-and-small-hv-study.md) | Travel Time Matrix and frozen Small H/V studies | Accepted implementation scope; engineering evidence is separate. | [0022](0022-composable-study-process-concurrency.md), [0035](0035-v3-persistent-dispatch-intentions.md), [0036](0036-v3-arrived-service-and-empty-reconsideration.md), [0037](0037-v3-service-and-learning-liveness.md) |
| [0022](0022-composable-study-process-concurrency.md) | Composable study process concurrency and truthful progress | User-authorized implementation scope; validation recorded separately. | — |
| [0023](0023-ray-tune-adaptive-execution.md) | Frozen experiment batches with measured adaptive execution | accepted for the authorized single-node implementation. | [0027](0027-balanced-performance-calibration.md) |
| [0024](0024-explicit-tasks-and-cooperative-stop.md) | Explicit CLI tasks and local cooperative stopping | User-authorized implementation; verification is reported separately. | [0025](0025-four-file-authoring-and-background.md), [0049](0049-case-level-evaluation-recovery.md), [0052](0052-unified-single-node-runtime-control.md) |
| [0025](0025-four-file-authoring-and-background.md) | Four-file authoring and explicit detached execution | User-authorized implementation; verification is recorded separately. | [0026](0026-four-file-only-authoring.md), [0049](0049-case-level-evaluation-recovery.md), [0058](0058-current-runtime-and-tracking.md) |
| [0026](0026-four-file-only-authoring.md) | Retire legacy author formats | User-authorized decision; migration and verification are in progress. | — |
| [0027](0027-balanced-performance-calibration.md) | Balanced and Performance calibration with frozen sampling layout | user-authorized implementation; engineering verification is separate. | [0028](0028-linked-preflight-and-terminal-control.md) |
| [0028](0028-linked-preflight-and-terminal-control.md) | Linked preflight, calibration and terminal control | user-authorized implementation; engineering and research evidence remain separate. | [0029](0029-three-level-runtime-progress.md) |
| [0029](0029-three-level-runtime-progress.md) | Three top-level runtime progress bars | user-authorized implementation; engineering display evidence only. | — |
| [0031](0031-v4-algorithm-matrix.md) | V4 Algorithm matrix axis and per-entry best ranking | Accepted for development experiments | [0058](0058-current-runtime-and-tracking.md) |
| [0032](0032-v4-directory-batch-execution.md) | V4 Experiment directory batches | Accepted for development experiments | [0033](0033-v4-performance-profiles-and-continuous-queue.md) |
| [0033](0033-v4-performance-profiles-and-continuous-queue.md) | V4 reusable performance candidates and continuous training admission | Development execution contract | — |
| [0034](0034-v4-explicit-calibration-skip.md) | Explicit V4 calibration skip | Development execution contract | — |
| [0035](0035-v3-persistent-dispatch-intentions.md) | V3 persistent Dispatcher intentions | Authorized implementation; verification reported separately. | [0036](0036-v3-arrived-service-and-empty-reconsideration.md), [0037](0037-v3-service-and-learning-liveness.md) |
| [0036](0036-configurable-experiment-contract.md) | Opt-in configurable experiment contracts | Development execution contract | — |
| [0036](0036-v3-arrived-service-and-empty-reconsideration.md) | V3 arrived service and empty reconsideration | Superseded by [ADR0037](0037-v3-service-and-learning-liveness.md). | [0037](0037-v3-service-and-learning-liveness.md) |
| [0037](0037-v3-service-and-learning-liveness.md) | V3 service and learning liveness | User-authorized repair; independent review corrections pending re-review. | [0038](0038-v3-blind-output-and-privileged-quality.md), [0053](0053-reservation-base-diagnostic.md) |
| [0038](0038-v3-blind-output-and-privileged-quality.md) | V3 blind OUTPUT and privileged shipment quality | User-authorized implementation; engineering verification separate. | [0053](0053-reservation-base-diagnostic.md) |
| [0039](0039-v3-fixed-window-task-reward.md) | Explicit V3 fixed-window task reward | User-authorized objective; engineering verification separate. | [0053](0053-reservation-base-diagnostic.md) |
| [0040](0040-online-experiment-concurrency.md) | Online experiment concurrency with reusable machine/task evidence | Development execution contract | — |
| [0041](0041-registered-public-observation-fields.md) | Registered public observation fields | Local development preparation; integration and independent review pending | [0045](0045-compact-algorithm-observation-inputs.md), [0046](0046-single-compact-public-interface.md) |
| [0042](0042-public-experiment-streams-and-stage-selection.md) | Explicit initialization, random streams and directory selection | Authorized local implementation; CI/integration review pending | — |
| [0043](0043-frozen-worlds-and-direct-case-streams.md) | Frozen worlds and direct case streams | Development execution contract | — |
| [0044](0044-training-passing-potential-with-terminal-closure.md) | Training passing potential with terminal closure | User-authorized training conversion; qualification pending. | — |
| [0045](0045-compact-algorithm-observation-inputs.md) | Compact algorithm-specific observation inputs | Authorized implementation; CI and integration pending | [0046](0046-single-compact-public-interface.md) |
| [0046](0046-single-compact-public-interface.md) | One maintained compact public-fields interface | User-authorized implementation; verification and integration pending | — |
| [0047](0047-live-manual-batch-queue.md) | Append-only manual queue around immutable experiment runs | user-authorized implementation; engineering verification is separate. | [0052](0052-unified-single-node-runtime-control.md) |
| [0048](0048-phase-execution-observations-and-estimates.md) | Phase execution observations and remaining-time estimates | accepted for implementation by the PR22 user request. | — |
| [0049](0049-case-level-evaluation-recovery.md) | Case-level evaluation recovery | user-authorized implementation; engineering verification is separate. | — |
| [0050](0050-production-progress-and-blocking-potentials.md) | Training production progress and blocking potentials | User-authorized local implementation; integration and formal runs pending. | — |
| [0051](0051-online-learner-shapes-and-stable-monitor.md) | Comparable learner seeds and stable monitor geometry | local implementation; runtime verification remains separate. | — |
| [0052](0052-unified-single-node-runtime-control.md) | Single-node execution control and logical task removal | user-authorized implementation; engineering acceptance is reported separately. | — |
| [0053](0053-reservation-base-diagnostic.md) | Reservation Base development diagnostic | User-authorized implementation; scientific acceptance remains separate. | [0054](0054-native-reservation-execution.md), [0058](0058-current-runtime-and-tracking.md) |
| [0054](0054-native-reservation-execution.md) | Native reservation execution and shared performance paths | User-authorized implementation; development verification only. | [0055](0055-reservation-lifecycle.md), [0058](0058-current-runtime-and-tracking.md) |
| [0055](0055-reservation-lifecycle.md) | Reservation execution lifecycle | User-authorized implementation; development verification only. | [0056](0056-base-elastic-and-case-recovery.md), [0058](0058-current-runtime-and-tracking.md) |
| [0056](0056-base-elastic-and-case-recovery.md) | Base elastic execution and case-level stage recovery | user-authorized implementation; verification evidence is separate. | [0058](0058-current-runtime-and-tracking.md) |
| [0057](0057-configurable-reservation-platform.md) | Configurable reservation experiment platform | user-authorized development integration; acceptance evidence is separate. | [0058](0058-current-runtime-and-tracking.md), [0079](0079-native-best-last-validation.md), [0080](0080-periodic-evaluation-naming.md) |
| [0058](0058-current-runtime-and-tracking.md) | Current runtime, explicit research modules and tracking | implemented; engineering acceptance and PR CI are recorded separately. | [0059](0059-explicit-algorithm-components.md), [0060](0060-monitoring-and-research-evaluation.md), [0061](0061-network-ownership-and-research-packages.md), [0062](0062-scenario-authoring-and-studio.md), [0063](0063-explicit-rule-actions-and-post-credit.md), [0078](0078-unified-operational-cli.md) |
| [0059](0059-explicit-algorithm-components.md) | Explicit algorithm components and action permissions | implemented; local engineering acceptance and review are recorded in [algorithm components validation](https://github.com/jian2028/SmartSOM/blob/5efd4d0bdde33c73a7528c3278a8307752a483cf/docs/validation/algorithm-components.md). Hosted CI remains a separate merge gate. | [0061](0061-network-ownership-and-research-packages.md), [0063](0063-explicit-rule-actions-and-post-credit.md), [0070](0070-central-rule-controller-components.md), [0073](0073-unified-observation-components.md) |
| [0060](0060-monitoring-and-research-evaluation.md) | Explicit monitoring and research evaluation protocols | accepted; implementation and final acceptance are tracked separately. | — |
| [0061](0061-network-ownership-and-research-packages.md) | Network ownership, fixed cases and research packages | accepted; implementation acceptance is recorded separately. | [0062](0062-scenario-authoring-and-studio.md), [0066](0066-four-map-catalog.md), [0070](0070-central-rule-controller-components.md) |
| [0062](0062-scenario-authoring-and-studio.md) | Human scenario authoring and Studio defaults | implemented; engineering verification and PR CI are separate evidence. | [0066](0066-four-map-catalog.md) |
| [0063](0063-explicit-rule-actions-and-post-credit.md) | Explicit action representation and POST pickup credits | implementation in progress; acceptance and hosted CI are separate evidence. | [0073](0073-unified-observation-components.md), [0076](0076-v2-rule-learning.md) |
| [0064](0064-incremental-batch-seed-selection.md) | Incremental batch execution selection | user-authorized implementation; independent review and CI pending. | [0065](0065-growing-batches-and-launch-order.md) |
| [0065](0065-condition-scenario-inputs.md) | Condition-specific scenario inputs | implemented in a proposed patch; independent review, integrated CI and formal experiment acceptance remain separate evidence. | — |
| [0065](0065-growing-batches-and-launch-order.md) | Append-only batch membership and execution order | inventory/stage implementation locally reviewed; the unpublished CLI ordering interface is superseded by ADR0066. Not yet published or deployed. | [0066](0066-bound-execution-seed-selection.md), [0078](0078-unified-operational-cli.md) |
| [0066](0066-bound-execution-seed-selection.md) | Bind execution subsets to ordered Experiment selections | accepted; implementation independently reviewed and locally validated. Publication and deployment remain separate actions. | — |
| [0066](0066-four-map-catalog.md) | Four standalone factory resources | implemented; engineering checks and independent review are separate evidence. | — |
| [0067](0067-trip-transit-time-ratio.md) | Dispatcher trip transit time ratio | accepted definition; implementation verification is separate. | — |
| [0068](0068-dqn-resident-packed-storage.md) | Lossless resident storage for built-in DQN rows | proposed in an isolated candidate; independent review, resource and end-to-end performance acceptance remain pending. No formal experiment is started or historical model migrated by this decision. | — |
| [0069](0069-cooperative-worker-resource-guard.md) | Cooperative native worker resource protection | implemented in a proposed patch. Integrated review, CI and the final 1280-tick post-start performance gate remain separate acceptance evidence. | — |
| [0070](0070-central-rule-controller-components.md) | Central rule controller through existing components | implementation; verification and formal experiments remain separate. | [0074](0074-custom-training-components.md) |
| [0071](0071-editable-training-seed-inventory.md) | Editable training seed inventory | development; engineering and independent review required. | — |
| [0073](0073-unified-observation-components.md) | One observation boundary and current experiment family | implemented; final review and verification are recorded separately. | [0074](0074-custom-training-components.md), [0075](0075-inline-study-panels.md) |
| [0074](0074-custom-training-components.md) | Common reward, critic and centralized learning contracts | implementation; verification is reported separately. | — |
| [0075](0075-inline-study-panels.md) | Algorithm-neutral scenarios and inline study test panels | implemented; verification is recorded separately. | — |
| [0076](0076-v2-rule-learning.md) | V2 rule learning and matched manufacturing scenarios | implementation in progress; no verification or experiment launch claimed. | — |
| [0077](0077-native-phase-resource-control.md) | Native phase resource control | implementation under engineering validation; no production migration. | [0078](0078-unified-operational-cli.md) |
| [0077](0077-v2-diagnostic-window-objective.md) | Explicit V2 diagnostic window objective | superseded for the consolidated diagnostic package by [ADR0081](0081-diagnostic-tail-provenance.md). Original diagnostic decision retained below; default V2 behavior remains unchanged. | [0081](0081-diagnostic-tail-provenance.md) |
| [0078](0078-unified-operational-cli.md) | Unified operational CLI | implementation under engineering validation. | — |
| [0079](0079-native-best-last-validation.md) | Bounded native checkpoints and V2 validation | accepted for implementation; experiment evidence remains separate. | [0080](0080-periodic-evaluation-naming.md) |
| [0080](0080-periodic-evaluation-naming.md) | Periodic evaluation naming | accepted for implementation by user request. | — |
| [0081](0081-diagnostic-tail-provenance.md) | Diagnostic tail provenance and explicit task-window comparison | opt-in development diagnostic, not a default learner change. | — |
| [0082](0082-native-training-sampling-controls.md) | Native sampling and per-network update controls | implementation under review; not a formal experimental result. | [0083](0083-explicit-target-clocks.md) |
| [0083](0083-explicit-target-clocks.md) | Explicit DQN target clocks and measured replay intensity | implementation under review; not an experimental result. | — |
