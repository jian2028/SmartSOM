# Reservation Base: integration audit before lifecycle changes

Audit date: 2026-10-25. Source: `codex/base-native-integration` over
`6bbf821` (PR27 integrated), including the previous uncommitted Base integration.
Canonical research inputs reviewed: `/Users/jianni/Documents/Jian/SmartSOM/plan/`
02–06 and README, plus subsequent explicit scenario decisions in this conversation.
The vault contains historical implementation snapshots and still-pending numeric
values; these do not override the later approved 4/4 capacities, 128/320 orders,
1,000 quality penalty and calibration starting at 2,000 ticks.

## Coverage

| Requirement | Current evidence before this change |
|---|---|
| Same core, five parameter groups, semantic IDs | ProductionSimulator Base contract, BaseProtocol, BasePPO; focused tests |
| Dispatcher station request; pre/post iterative acceptance + STOP | Implemented; no STOP in on-site service or Input acceptance |
| Reservation versus physical capacity; 1 tick service | Implemented with service locks and core assertions |
| Nominal static distances, released-order observation, CTDE public critic | Explicit role feature construction; hidden-information tests |
| Mode/quality, final Output termination, gamma=1, reward -(D+lambda F)/N | Implemented; telescoping checks on complete episodes |
| Finite arrivals, normal/low/medium/high, conditional truncated exponential | Implemented; existing calibration candidate is not automatically frozen |
| Shared fixed Mover | Implemented with one-step joint collision checks and displacement |
| General moving-trajectory event skipping | Partial: idle jumps exist; moving vehicles still advance one grid interval |
| Full paper algorithm/model selection | Not complete: PPO diagnostic only; value-learning comparison and prescribed best-checkpoint selection remain separate |
| Background / durable continuation / Tune | Not connected to Base before this change |

## Performance actually on the Base path

Merged code is not proof of hot-path execution. Base bypasses the old collector
and feature builder. Equivalent avoided-next-value work and direct typed vectors
already exist. Remaining-work caching and static-route caching execute in Base.
Actor-only evaluation and per-update raw tensor reuse were explicitly adapted and
checked for trajectory / parameter equality. Prior 500-tick matched tests measured
1.16–1.22x evaluation speed and 1.62–1.83x update-stage speed; these are not whole
training speedups. Old DQN-only state-digest work is not relevant to Base PPO;
old model-packing caches cannot be copied blindly across changed weights.

## Structural defects and remedies

1. `config.reservation` writes temporary YAML to reuse file loaders. This conflicts
   with ADR0025's detached compilation contract. Extract a real in-memory parsing
   path shared with file loading; persist YAML only as execution evidence.
2. `last.pt` has model/optimizer/RNG but no committed episode, condition RNG,
   budget or log transaction. It is not a resumable experiment by itself. Add one
   complete-update continuation owner, atomic generations, source/input checks,
   durable deadline and idempotent progress/log recovery. Interrupted uncommitted
   rollout is replayed from that boundary, not falsely labeled mid-tick resume.
3. Native foreground and legacy diagnostic launch should share the same training
   session. Avoid separate background training loops or Tune-specific algorithms.
4. Existing `background.launch` already verifies PID birth, closes stdin, owns
   logs, and handshakes with the child. Extend its schema dispatch; do not add a
   second daemon or PID-only kill mechanism.
5. Existing resource monitor, bounded probe supervisor/calibration controller and
   Tune adapters provide reusable execution mechanisms. Add the Base session
   adapter while preserving the simulation/learner contract; keep Ray optional.
6. Duplicate JSON writing and fabricated argument/loading paths should be reduced
   where the common project components already provide the exact behavior.

## External implementation references

[Ray Tune fault tolerance](https://docs.ray.io/en/latest/tune/tutorials/tune-fault-tolerance.html)
separates experiment restoration from trial checkpoints and bounded failure
retries. A restored driver alone cannot reconstruct omitted learner state.
[Ray Tuner.restore](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.restore.html)
also distinguishes resuming errored trials from restarting them from scratch.
The repository pins Ray 2.58.0; its actual local API is the implementation target.

[Stable-Baselines3 save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)
keeps optimizer, model, algorithm metadata and environment/version information.
This supports explicit continuation contents rather than treating weights as the
entire run. Neither framework is evidence that SmartSOM's physical state or
scientific budgets are automatically preserved.

## Scope of the following implementation

Connect execution/background/continuation/Tune, fix detached compilation and
reuse existing lifecycle/resource utilities. Do not change rewards, legal STOP,
Mover, actor observations, rollout return targets or paper algorithm selection.
Do not upgrade partial Plan coverage into a claim that all research is complete.
Do not rewrite the vault or old run/model artifacts. Verification must include
actual detached processes, interrupted/resumed equivalence, completed-stage
idempotence, identity drift rejection and a real optional Ray run.

## Implemented after this audit

- Removed temporary-file compilation; shared Factory parsing accepts detached data.
- Native and legacy Base training now share ReservationSession and atomic update
  generations. Existing background handshake, common JSON evidence, checkpoint sync
  helpers and resource/probe machinery are reused.
- Optional Ray uses the same learner session. Its independent-seed fixed CPU profile
  is measured before launch; live elastic rescheduling is deliberately not advertised.
- Shared stop and continuation were tested with a real detached Ray driver: two seeds
  resumed their committed counters and finished10000 ticks each without deadline reset.
- Ray reporting checkpoint frequency is10 iterations; the authoritative native state
  still commits every rollout, avoiding redundant Ray state snapshots at each short
  rollout. Native generations are currently retained for audit; long-run retention
  policy remains a storage follow-up, not a scientific contract change.
- Unfinished evaluation/calibration stage recovery, live elastic/GPU execution,
  full moving-trajectory jumps and value-learning/model-selection research remain
  outside this completed training-lifecycle adapter.

Local evidence lives in `artifacts/base-runtime-audit/REPORT.md` (ignored generated
artifacts). Full supported Linux/macOS CI remains the merge gate, not a claim made
from local smoke runs. No old server experiments or Obsidian Plan files were changed.
