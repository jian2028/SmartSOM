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

## Remaining full acceptance

The approved main acceptance freezes the first four sorted S00 jobs (nine
operations), changes only the declared learning arrival profile (two initially
available, others uniform [1,20], reveal=release), and generates quality per
episode. RLlib PPO uses 4096 steps; SB3 MaskablePPO uses 1024 with rollout 256 and
batch 64. Root seed is 101. Both need finite updates and restored checkpoints.
Evaluation uses study root 202 and replications 0–4, paired with SPT, for 15
completed runs and exact action/schedule replay. Budgets, seeds and physics may
not be changed to turn an unsuccessful acceptance into a success.

Training, checkpoint and formal evaluation results are not yet established by
this interface checkpoint. Actual Linux execution remains unverified.
