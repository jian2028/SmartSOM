# Batch 01 · Small H/V development pilot

**Status:** completed development pilot, not a formal multi-seed research result. The source was dirty when prepared; the 24 conditions and five held-out cases per condition completed without evaluation exceptions. [Frozen study](/Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM/runs/prepared/small_hv_workflow/plan.json) · [best-checkpoint selection](/Users/jianni/.codex/visualizations/2026/09/27/01a0e48e-830b-7f03-87b3-e79b8ce2c500/best-checkpoint-comparison/selection.json) · [full comparison](/Users/jianni/.codex/visualizations/2026/09/27/01a0e48e-830b-7f03-87b3-e79b8ce2c500/best-checkpoint-comparison/comparison.csv).

Small uses eight machines, eight AGVs, 64 finite orders and a 4,096-tick episode limit. The 24 conditions cross H0/H1, three V levels, PPO/DQN, and automatic/zero travel time. Each condition trained for 16,384 physical ticks with one training seed. Checkpoints below were selected by complete validation cases, mean deliveries, raw return, then earliest update; the original final evaluation instead used `last`.

| Travel time | PPO best: deliveries / 64, complete cases / 30 | DQN best: deliveries / 64, complete cases / 30 |
| --- | ---: | ---: |
| Automatic | 58.73, 19 | 41.13, 4 |
| Zero | 45.53, 19 | 63.57, 22 |

Two targeted failures motivate [Batch 02](batch-02-plan.md): `h1_low_ppo_zero` best update 4 delivered 0/64 in every test case; `h0_high_dqn_auto` best update 60 averaged 25.4/64. In the latter's case 3, all eight empty AGVs chose `NO_REQUEST` for 1,648 consecutive ticks despite legal pickup targets; in cases 0, 2, and 4, loaded AGVs also waited at full destination buffers. A same-condition `h1_low_dqn_zero` best update 24 averaged 63.4/64, but its three 63/64 cases also ended with long all-idle periods and legal pickups. These trace observations motivate distinct missed-dispatch and congestion tests; they do not prove a causal role or establish an algorithm ranking across seeds. The finite test set has a hard upper bound of 64 deliveries per case; a throughput estimate for unlimited backlog is a different quantity.
