# 0065 — Condition-specific scenario inputs

Status: implemented in a proposed patch; independent review, integrated CI and
formal experiment acceptance remain separate evidence.

Extends ADR0062's Experiment input selection and ADR0061's fixed-case provenance.
The single-scenario default, physical generation laws and learning algorithms are
unchanged. This is limited to the existing four fault conditions.

An optional `condition_scenarios` mapping supplies exactly normal, low, medium and
high scenario paths. All mapped scenes must have the same expanded factory and
job-generation settings as the primary scenario. Only machine breakdown settings
may differ. Files may share contents or paths; severity comes from the existing
condition argument and scene profile dictionary, never the filename. The primary
scenario remains required, including for legacy expanded
factory/workload callers. Controlled recovery scenarios are excluded.

Compilation resolves all four paths before execution, preserves source hashes,
freezes their authoring documents and expanded factory/workload evidence, and
includes all four resolved inputs in scientific identity. Equivalent authoring
representations do not change that identity; raw file hashes still bind exact
continuation provenance. The content-keyed parsed-scene cache is bounded and does
not appear in serialized scientific inputs. Each lookup returns a deep copy so
nested mutable fields cannot contaminate another binding or future load.

Each training episode selects the already parsed inputs after the existing
condition selection. One model and its RNG state span all conditions. Neither
world seeding, update cadence nor budget semantics change. Evaluation, calibration
and completion-horizon generation select the same inputs by case condition.
Continuation consumes frozen paths and rejects source/input changes through the
existing snapshot and manifest checks.

Mapped fixed-case files and recipes additionally declare
`condition_inputs_sha256`, with a factory and workload digest per condition.
The top-level factory/workload identity remains bound to the primary scenario.
Imported physical worlds must reproduce from the selected frozen condition inputs
and their declared seeds. Their case IDs, action seeds and inventory digest remain
unchanged by materialization. Comparison and reference identities also include the
condition inputs, preventing different fault settings from appearing to be an
observation-only ablation.

No new experiment driver, general scenario-provider framework, fault-condition
model expansion or automatic launch is introduced. Configuration and bounded
engineering checks establish this wiring, not scientific results or performance.

The checkpoint sidecar also binds an independent, self-digested training context.
Evaluation retains that context separately from its own inputs and allows OOD
worlds. Strict paired algorithm comparisons require equal training provenance;
legacy evidence without it cannot establish pairing. Dashboard cohorts include
all frozen condition inputs and checkpoint training contexts, and ambiguous
same-seed independent runs do not silently replace one another in statistics.
