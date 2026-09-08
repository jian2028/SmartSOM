# Centralized learning acceptance

Item 12 has two independently verified checkpoints. The shared projection/Gym
interface does not establish learner training or integrated acceptance.

## Shared interface

The focused suite covers every legal combination of arrivals, breakdowns,
processing uncertainty, AGV, finite buffers, quality and holding (holding requires
AGV), with both arrival triggers. Every SPT-selected semantic action is encoded
and decoded without a framework-specific simulator; the full trace equals direct
core execution, and action/schedule replay retain physical results and quality.

Additional cases cover the static FJSP hand input, reveal-order binding, hidden
job counts/attributes and quality probabilities, strict expanded-mode capacity,
semantic ordering, immutable snapshots, public waiting witnesses, initialization
time, invalid-action atomicity, actual buffer deadlock, `policy_stalled` with a
hidden future arrival, zero-time rerouting and post-transition budget limits.
Both plain and masked views run the official Gymnasium 1.2.2 checker. Its infinite
Box-bound advice is expected: values are checked for finiteness, not clipped to
an episode-derived range.

```bash
uv sync --locked --extra cp --extra gym
uv run --no-sync pytest -q tests/unit/test_learning_projection.py tests/unit/test_learning_gymnasium.py
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
```

Local development gate logs belong in `artifacts/learning/item12-checks/`. They
identify a working-tree check, not the final integrated training experiment.

The first checkpoint's macOS development gates completed with **310 focused
checks**, **1162 full CP+Gym checks**, and **995 base checks** (12 optional CP
cases and the optional Gym test module skipped in the base environment).
The CP suite actually solved and replayed ft06=55, official FJSP=6 and Mk01=40.
Ruff, formatting, lock consistency, installed dependency consistency and diff
whitespace checks passed. CI now requires the Gym extra explicitly; Ubuntu
execution has not been observed locally. These are interface engineering results.

## Training and checkpoint acceptance

The approved main acceptance freezes the first four sorted S00 jobs (nine
operations), changes only the declared learning arrival profile (two initially
available, others uniform [1,20], reveal=release), and generates quality per
episode. RLlib PPO uses 4096 steps; SB3 MaskablePPO uses 1024 with rollout 256 and
batch 64. Root seed is 101. Both require finite updates and restored checkpoints.
Evaluation uses study root 202 and replications 0–4, paired with SPT, for 15
completed runs and exact action/schedule replay. Budgets, seeds and physics may
not be changed to turn an unsuccessful acceptance into a success.

The micro fixture is `data/reference/learning/s00_micro.json`, with its frozen
selection and source digests in `source.json`. The scenario preserves the original
eight machines, four AGVs, matrix, buffer/holding capacities and quality table.
Its training arrivals are a new, explicitly declared acceptance profile, not
an IDETC numerical reproduction. No SPT logistics admission filter is applied to
the learner's physical action mask.

Locked versions are Gymnasium 1.2.2, Ray/RLlib 2.58.0, SB3 and sb3-contrib 2.9.0,
and Torch 2.14.0. Both use one CPU environment, one numerical thread, a 64×64
tanh MLP and gamma 1. RLlib's 4096 steps are 16 rollouts/training iterations of
256; SB3's 1024 steps are four rollouts of 256. SB3 reports `learner_updates=40`
because that counter counts its ten optimization epochs per rollout, not forty
independent sampling iterations.

Preparation tests cover strict training/algorithm/budget schemas, immutable input
materialization, generated-base freezing, fixed imports, backend-independent episode
seed goldens and RNG isolation. Checkpoint tests reject missing dependencies, corrupt
members and mismatched structure/projection before Simulator/directory creation.
Validation creates no run directory, and `run` does not implicitly train. Failure
tests retain partial traces and the original writer/provider cause; nonfinite
learner metrics abort instead of being sanitized or replaced.

Real integration tests train both fixed budgets, compare initial/final weight
digests, verify restored weights, audit every recorded episode and run the full
15-entry evaluation study. `SMARTSOM_REQUIRE_LEARNING=1` makes absent learning
extras a failure. Base CI can skip optional integration tests; that is not learning
acceptance. The full macOS learning+CP gate completed **1183 tests**, including
actual ft06=55, official FJSP=6 and Mk01=40 solve/replay regression.
The independent base gate completed **1011 tests** with 18 explicit optional
skips; no learning/solver packages are installed in that environment. Ruff,
formatting, lock and installed-dependency consistency, and diff checks passed.

## Development result and formal protocol

The development attempt on the first checkpoint plus the uncommitted second
change completed both budgets and all 15 evaluations. Its local report is
`artifacts/learning/development-acceptance-1/report.json`. RLlib completed 116
episodes and SB3 29, each with one recorded partial episode at sampling cutoff;
neither had a failed completed episode. All 5120 sampled decisions replayed with
matching semantic actions, rewards, observation/mask digests and trace digests.
Successful training episodes and all evaluation runs also passed full schedule
replay. Evaluation preserves ordinary paired study inputs across the three
algorithms, including the same quality draws for each replication.

| Algorithm | Development mean makespan, five evaluations |
| --- | ---: |
| SPT | 134.6 |
| RLlib PPO | 449.2 |
| SB3 MaskablePPO | 595.8 |

These short-trained models are slower than SPT on this fixture. The engineering
gate does not require beating SPT, learning-curve convergence or a statistical
performance claim. Failed development attempts remain local evidence, including
fixed-version Ray integration diagnostics; they are not counted as successful
training. No seed, budget or physical input was changed to obtain this result.

After the second commit, run the following from that integrated main source:

```bash
uv sync --locked --extra cp --extra learning-rllib --extra learning-sb3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run --no-sync smartsom train configs/runs/learning_rllib.yaml
uv run --no-sync smartsom train configs/runs/learning_sb3.yaml
uv run --no-sync python scripts/validate_learning.py \
  --rllib-training-dir PATH_FROM_FIRST_TRAIN \
  --sb3-training-dir PATH_FROM_SECOND_TRAIN \
  --output-dir artifacts/learning/formal-NEW_COMMIT --workers 2
```

The shell controls only execution resources and artifact references. Committed
run/study configuration owns scientific parameters and seeds. The acceptance
script fills model/output paths in `configs/studies/learning_evaluation.yaml`;
it does not override the fixed world, algorithms or budget. It audits training
before batch execution and preserves failures. `report.json`/`report.md` link
attempts and record input pairing, per-run metrics, replay results, actual source
and dependencies. The main-commit report must show both exact training budgets
and **15/15 complete, replayed evaluations** before item 12 is complete. A
successful batch process alone is insufficient. Generated runs are not committed.

Ubuntu CI includes the same real training and paired replay gate. No actual Linux
execution has been observed in this local, no-push task; its status is pending.
