# Runtime consolidation engineering acceptance

Scope: one current physical/runtime implementation, operational profiles and
launch/control/discovery, native Base/SL PPO/DQN, centralized assignment, local
metric events, W&B/TensorBoard, and historical evidence access. No formal research
experiment or algorithm superiority claim is made.

Baseline: `c6aa68e5609de51f0ef50ab8c37d1bb794237693`.
Current Python implementation SHA256:
`8ceb54f922dcf189528f43a526d15ccf76608cc41fa373b1d95739ca6b6bfc5e`.
This hashes sorted package-relative Python paths, NUL separators and source bytes,
as produced by `reservation_runtime.source_identity()`.

## Independent review

Three independent read-only reviewers examined engineering integration, scientific
semantics/reproducibility and validation/dependency coverage. They received the
same candidate manifest including untracked files and deletions. Subsequent
source deltas were independently rechecked; intermediate reviews are not used as
blanket acceptance of later edits.

Confirmed findings repaired and independently closed:

- Cooperative worker pause/stop no longer becomes a failed process exit.
- Batch continuation reads operation overrides before first scheduler setup and
  propagates explicit changes to appended children; live runtime limits cannot
  defeat a subsequent explicit resume override.
- Historical batch control commands reject mutation through all public aliases.
- Selected numerical threads are installed before session restoration/validation.
- DQN validation timing includes learner updates using the same metric definition.
- Lean CI preflight declarations match installed extras.
- Historical spatial playback retains a public, read-only, seed-aware route.
- Evaluation checkpoint fingerprints exist before tracking initialization, and
  appended trials share the parent tracking group.

Independent scenarios included manual DQN reward intervals with zero-time stages,
hidden future/quality perturbations, fresh-process RNG isolation, two additional
Base trace comparisons, and guarded no-framework CLI→smoke→ID/results→matrix runs.
These are engineering evidence, not scientific results.

## Verification

- Ruff lint and format, Git whitespace checks and full ownership-plan validation.
- Corrected Base reference: six normal/high seeded traces and two additional
  medium/high/EMPTY_WAIT/STOP combinations match the frozen baseline's decisions,
  events, actions, time and reward.
- Real native four-combination learning/update/checkpoint contracts, custom
  observation/network freezing, centralized capacity/ownership/failure checks.
- Real W&B 0.30.0 offline segments and TensorBoard events; Python/NumPy/Torch RNG
  equality. Mocked online startup/resume/upload failure behavior is separate from
  an actual cloud connection, which has not been verified here.
- Real Ray pause releases the actor, preserves native continuation and reaches
  its original 1,000-tick engineering budget after resume. Real native worker
  pause/resume advances its checkpoint and exits cooperatively on stop.
- Current and historical read-only playback checks, including byte preservation.
- An independently installed minimal environment contains no Torch, Ray,
  Gymnasium, W&B, TensorBoard or Qt; current CLI and fast contracts execute there.
  The final fast ownership slice passed **951 tests**, with two platform skips,
  in that environment (91.73 seconds).

The final full local suite passed **1,550 tests**, with four platform skips, in
390.40 seconds on macOS/Python 3.12.13. The command used the locked environment
with learning, CPU, tuning, Gymnasium, W&B, TensorBoard and Studio extras:

```sh
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 WANDB_MODE=disabled \
  PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  uv run --no-sync python -m pytest tests -q --maxfail=5 -p no:cacheprovider
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-project python scripts/ci/selection.py --event workflow_dispatch
git diff --check
```

Hosted CI results are reported by the PR checks. A local macOS run is not
evidence of hosted Linux/macOS success. CUDA/H20, native Windows
and real online W&B synchronization remain outside this measured acceptance.

## Short-trajectory performance

Same machine and Python environment, three alternating baseline/current processes,
six fixed rule trajectories per process (1,379 total physical ticks), including
interpreter startup. Every normalized trace hash is identical. macOS `/usr/bin/time
-l` supplies peak RSS. No other acceptance suite ran during the measurements.

| Measure | Baseline | Current |
| --- | ---: | ---: |
| Median seconds | 1.4513 | 1.3386 |
| Median physical ticks/s | 950.2 | 1,030.2 |
| Maximum RSS across three processes | 86,179,840 B | 78,659,584 B |

Observed throughput is 8.4% higher and RSS 8.7% lower in this bounded workload.
These measurements do not establish general training speedup or convergence.

## Archive boundaries

Both source tar checksums and 139 retired-test entries were independently verified.
A representative older PPO component loaded metadata, encoder and 16 weight tensors
under its matching preserved dirty source. The newer pre-consolidation source
correctly rejects it. No evaluation or complete continuation is inferred from this
weights-only check. See [archive reconstruction and levels](../archive/README.md)
and [the exact index](../archive/index.json).
