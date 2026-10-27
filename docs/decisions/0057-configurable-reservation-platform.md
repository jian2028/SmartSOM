# 0057 — Configurable reservation experiment platform

Date: 2026-10-27
Status: user-authorized development integration; acceptance evidence is separate.

Extends ADR0053–0056 without changing the default native reservation physics,
reward, ordering, Mover, sampling, PPO grouping or physical-tick budget. Explicit
research variants are scientific inputs, not transparent optimizations.

Algorithm owns role.stage bindings, declared Python provider sources, observation
version, STOP/empty WAIT permissions and learner target/initialization scales.
Experiment owns tick or episode budgets, budget-cut behavior, condition sampling,
validation milestones/selection, periodic checkpoint exports and diagnostics.
Fully expanded typed inputs and override provenance are frozen before execution.
Unknown fields fail validation; Python/YAML loading is outside the rollout loop.

Actual binding points are machine.process, dispatcher.pickup/delivery,
pre.accept/service, post.accept/service and input.accept/service. Output service
remains automatic. A provider receives detached public numeric decisions and
semantic candidates, returning a candidate identity. It never receives the core.
Its source digest/factory participates in checkpoint compatibility; execution
freezes the standalone source file. This is a Python extension interface, not a
security sandbox. Parameter/optimizer ownership stays in the existing five-group
PPO; nonlearning decisions do not produce PPO records or critic encodings. Return
suffixes still include all physical rewards across intervening rule decisions.

native-v1 preserves the dispatcher field-18 overwrite for historical evidence.
native-v2 retains the next-operation one-hot at [18:22] and puts the rejection
count at 22. Both have width 48. Neither is frozen server C or reconstructed Mac
encoding; their checkpoints are not silently imported. Empty WAIT is separately
opt-in, only for uncommitted empty vehicles with candidate targets; it prevents a
repeat decision until the next physical tick. The Mover remains unchanged.
Enabled WAIT has an explicit dispatcher head; legacy default model keys/RNG order
are retained when disabled. STOP remains default-on at pre/post accept only.

Episode budgets count committed complete rollouts, including scientifically
truncated drain episodes; wall fragments do not update or consume an episode.
Random condition sampling remains the default; balanced_cycle is explicitly a
new variant. Tick budgets cut at the exact limit by default; complete_episode can
overshoot to the next episode boundary. Learning reward scaling affects targets,
not physical reward or reported joint cost. Critic initialization scaling is an
independent explicit setting and does not reproduce every server initialization.

Validation uses the same evaluator and CaseStore as public evaluation. All
configured panels must finish before selection. sample_complete_cost sorts by
incomplete sample cases, then joint cost only when all complete, then earliest
episode. sample_delivery_cost adds delivered fraction before cost. Inclusion of
episode zero is explicit. Neither mode alone establishes full reproduction of
server/Mac experiments; their world seed namespaces and encodings differ.
Default validation is disabled and default selection is last. Validation restores
Torch global RNG on exit and uses independent evaluation policy objects.
Periodic model exports are separate from every-update atomic recovery generations;
this integration retains native recovery history rather than silently pruning it.

Base directory inputs reuse author_batch and NativeScheduler, including admission,
cooperative controls, child identity checks and failure propagation. Batch prepares
native single-session children; standalone Tune retains shared adaptive broker and
Ray sessions. `run.resource_execution` exposes the existing broker fixed/adaptive
execution modes (adaptive by default), without adding a learner or scheduler.
Fixed mode retains initial admission and cooperative controls, but preserves the
calibrated worker allocation after admission. Mixed Base/V4 directories and nested
Tune children are rejected.
This does not add a separate scheduler, episode loop or distributed learner.
