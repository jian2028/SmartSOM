# 0079 — Bounded native checkpoints and V2 validation

Date: 2026-11-08
Status: accepted for implementation; experiment evidence remains separate.

Supersedes only ADR0057's unconditional native recovery-history retention for
experiments opting into `checkpoint_retention: best_last`. The default `all`
continues retaining history. Existing frozen runs are not migrated.

Every committed update publishes a complete atomic continuation generation,
including learner/optimizer/replay/RNG state and episode records. After publishing
the current pointer and last alias, only older owned generations are collected.
Unrecognized or foreign directories are preserved. The best model is independently
retained. Immutable model aliases use hard links where available, otherwise copies;
publication requires temporary storage and is not a strict two-file disk bound.

Bounded retention requires validation selection. Completed validation panels keep
scores, receipts and checkpoint provenance, but release their model copies after
best publication. Resume reuses completed panels and verifies a previously selected
best against its retained provenance. Incomplete panels retain their model for
case-level recovery. Earlier prescribed final-test checkpoints are rejected because
they would no longer be retained. The final prescribed episode evaluates last.

V2 Base/ISL shared/independent and Morris recipes use milestones50..1000 in steps
of50, with12 fixed validation cases (three per normal/low/medium/high condition).
The panel is shared across learning methods and training seeds within each scale,
separate from training and the80 pinned final test cases. Argmax selection ranks
incomplete case count, then mean joint cost only for fully completed panels, then
earlier milestone. This extends the existing complete-cost rule to argmax without
using censored complete-case means. Validation isolates Torch RNG from training.

Last remains a complete continuation checkpoint at every committed episode,
not merely every50 episodes. Best is absent before the first eligible validation.
Final testing remains the prescribed episode1000 model; best retention does not
silently change the final-test selection protocol. These engineering contracts do
not establish statistical adequacy or algorithm superiority.
