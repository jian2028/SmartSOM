# Frozen social-learning development comparison

`python -m smartsom.experiments.social_long` is an isolated POSIX entrypoint for
this approved development run. It reuses `base_diagnostic.episode` and `BasePPO`,
without changing their scientific implementation. A frozen feature commit is not
formal main/tag research evidence. Existing server/Mac results are not this Base.

The production manifest fixes nine arms (`base`, `machine`, `dispatcher-pickup`,
`dispatcher-delivery`, `dispatcher`, `pre`, `post`, `input`, `all`), seeds
4101/4102/4103, and 1000 complete rollout/update episodes per run (27000 total).
An episode ends at completion or the common drain deadline; it is not cut to hit
the training tick threshold. Target factory/workload/algorithm files provide
320 orders, 20 machines/AGVs, pre/post capacity four, drain 2000, width 32,
quality penalty 1000 and unchanged PPO. Overrides common to every arm are
reconstructed inputs, reservation STOP and empty WAIT. Loaded active WAIT remains
absent. Only actor social branches vary; pooling is attention. Learner rewards
are divided by 1000. Each seed starts fresh; no old weights are loaded.

Training worlds depend on training seed and episode, with normal/low/medium/high
conditions in rotation. World and action seed domains are separate; evaluation
worlds/actions pair across groups and training seeds. Validation uses eight worlds
(two per condition), sample twice and argmax once, at episodes
0/100/200/300/500/750/1000 and early-160k. After all training and validation finish,
test uses the same 32 worlds for initial, early-160k and episode-1000 policies.
There is no test-based selection and no automatic validation-best test budget.
Identical checkpoint hashes can reuse a panel only within the same arm/seed,
split and decoder, with an explicit `reused_from` link. Rule/random are shared
controls: all arms have identical action/protocol options, and these policies
ignore social observations. Their shared results are not nine independent
replications. Initial neural evaluation is retained separately for every run.

Each full episode PPO update commits `last.pt` atomically with model, Adam,
policy/Python RNG, Torch generator and global Torch RNG, episode/tick/update
counters, accumulated active training elapsed seconds and the authoritative row.
The directory and file are fsynced. Before the next update (also on recovery),
reconciliation repairs the ledger's trailing incomplete record, rejects malformed
complete records/conflicts, normalizes tuple/list JSON representations and
publishes required immutable snapshots. Every 50 episodes is retained.

The first post-update cumulative tick >=160000 saves `early-160k.pt` with actual
tick, episode, overshoot and SHA256 receipt. A crash after `last.pt` but before
its early copy or ledger is recovered before training resumes. Later states
cannot substitute for a missing earlier crossing. `last.pt` and early snapshot
have identical bytes at crossing, and training continues from that same state.

The independent controller locks its runroot; children inherit the lock so an
orphaned finite worker blocks a competing controller until it exits. Each run
also has its own worker lock. Ten-episode blocks advance in full-panel waves;
no seed is filtered by score. Worker errors create a visible failure receipt,
drain other finite workers and stop admission; restarting does not retry the
scientific error automatically. Diagnose it explicitly before further operation.

The latest authorized shared-host cap is 28, superseding the initial 24 proposal:
reserve the old Base controller's 24 promised slots while it lives, and admit at
most four SL workers. Never change its scripts, deadlines or running source.
Admission also requires >=16 GiB available, measured CPU <85%, >=2 idle logical
CPU equivalents and room after external Python/Ray workers. SL processes run at
nice +10 with one Torch/BLAS thread. This is conservative observation, not a
shared lock with other schedulers; it cannot guarantee unchanged Base throughput.
External workloads remain visible in status. No automatic growth beyond four is
implemented; raise capacity only after a separate justified coordination change.

Use the existing runtime without synchronizing/removing shared dependencies:

```sh
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
uv run --no-project --python /home/jianni/code/SmartSOM/.venv/bin/python \
  -m smartsom.experiments.social_long --root /home/jianni/runs/SL_RUN init
uv run --no-project --python /home/jianni/code/SmartSOM/.venv/bin/python \
  -m smartsom.experiments.social_long --root /home/jianni/runs/SL_RUN coordinate \
  --legacy-root /home/jianni/runs/base-server-resumed-20261027
```

Production initialization requires a clean commit, snapshots source/config hashes
and dependencies, and refuses an existing output directory. Workers verify frozen
hashes and the imported source path. `init --engineering` uses the small test
recipe, 20 episodes, threshold one and drain 50 in a separately marked runroot;
it cannot be mistaken for a production budget. Do not use engineering evidence
as the 27-run comparison. It exists for real multi-update/process recovery checks.

`status.json`, per-run `progress.json`, launches and worker logs show live facts.
`report` and the controller produce all-seed CSV panels and HTML curves on
training episode, physical tick and active elapsed-time axes. Full-panel J is
reported only when every trajectory drains; delivery fraction keeps failures
visible. Missing evaluation is not zero and successful subsets are not ranked.
Model, logs, local review receipts and results stay out of Git.
