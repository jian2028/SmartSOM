# Local reconstructed Base long-training diagnostic

This isolated, uncommitted worktree is a **development experiment** based on
253c818. The remote STOP×WAIT patch and checkpoints were unavailable during the
BudgetVM outage. It is not an exact reproduction of the remote frozen source.
No production checkout, old model, reward or Mover is changed.

## Frozen comparison

A: reservation STOP on, empty Dispatcher WAIT off.
B: STOP off, WAIT off. C: both on. D: STOP off, WAIT on.
Loaded active WAIT remains unavailable. Input acceptance and on-site service have
no new STOP. An empty vehicle without reservation/service can choose EMPTY_WAIT,
including when it is the only action. WAIT exits all remaining negotiation rounds
of the physical tick, wakes next tick, consumes time and keeps actual occupancy.
Without WAIT the original opportunity/retry wake rule applies. STOP off continues
serial acceptance until quota or candidates are exhausted, not simultaneous overbooking.

The all-empty WAIT test and physical one-tick advancement are explicit gates.
WAIT does not prevent the unchanged shared Mover from yielding an idle vehicle.
All arms have identical network architecture and paired initial parameters.
The WAIT head is present in each role network, but selected only for Dispatcher
EMPTY_WAIT actions; unused heads are not evidence of learning.

## Reconstructed C input semantics (48-wide rows)

Original fields are preserved except the documented Dispatcher collision.
Dispatcher rejected count moves from context[18] to context[22], preserving
following operation type one-hot at [18:22]. Station rows and their first-row
context add current remaining standard-mode work / time_scale at [25].
Dispatcher destination candidates add the corresponding remaining work at [44].
Post context retains the previous station context and adds **its own** pickup
quota at [26] and commitment count at [27], both / count_scale. Other relational
encoding weaknesses are not repaired in this baseline. These slot assignments
are the local reconstruction schema, not a claim about remote byte identity.
Current work uses public elapsed fraction and nominal standard duration, zero
when not processing; it does not reveal future faults or latent quality.

## Training and evidence

Target scene only: 320 orders, 20 machines/vehicles, pre/post4, drain2000. Four
arms × three new paired seeds3101/3102/3103, each targets1000 episodes. The paired
scenario namespace and uniform normal/low/medium/high cycle are fixed before
results. Gamma1, original role-per-episode advantage normalization and PPO
options remain unchanged; reward targets divided by1000, physical reward and
reported J unscaled. No terminal penalty or deadlock shield is added.

Chunks of10 episodes permit fair scheduling across12 runs at8 initial compute
slots. Checkpoints include model, optimizer, sampling/shuffling RNG, Torch RNG,
counts and the last episode row. Last checkpoint is atomic every complete
update; models retained every50 episodes. A mid-episode interruption restarts
that episode from the last complete checkpoint; it is recorded, not called
uninterrupted execution. Complete restored-state equivalence is tested locally.

Validation at0/100/200/300/500/750/1000 uses8 fixed worlds (2 per fault level),
sample2 repetitions and argmax1. Final test32 independent worlds; checkpoint
selection uses completed sample validation panels lexicographically: drained
fraction, delivery fraction, complete-panel J only when all drained, earlier
checkpoint tie-break. Initial networks and same-action-space rule/random get
separate test panels. Test never selects checkpoint. Random samples all legal
job×mode/STOP/WAIT choices. Rule keeps old ordering and avoids voluntary WAIT
when a source is available. No cross-arm baseline reuse.

The local run retains the already authorized absolute deadline1793144300.769166
(2026-10-28 07:38:20 Asia/Shanghai), training cutoff1793129900.769166. It does not
silently restart a36-hour budget. At cutoff, completed checkpoints and partial
evidence remain. Missing evaluations or fewer than1000 episodes are reported.

## Operations

Use `uv run --no-sync --project /Users/jianni/code/SmartSOM python` with
`PYTHONPATH` pointing to this worktree's `src`. Existing Torch runtime is reused;
source imports and full source/config hashes are verified by each worker.
`run.py --root <new-path> init` freezes source and manifest. Then `coordinate`
controls registered children and records PID/command/exit codes. Only one
coordinator and one worker per training/evaluation directory may hold its lock.
`control.json` adjusts future queue admission (workers, pause_admission), never
scientific semantics of running workers. No automatic failed-task retry.

Performance copies use the same48 target-scene/seed/arm tasks per concurrency,
1024 physical ticks and real PPO update. This measures early rollout+update
throughput, not full-run throughput; confirm memory and real long-run throughput
before further increases. Start8, test12/16 if useful, upper bound18 logical
cores, leaving headroom when thermal/memory pressure or throughput regression
appears. Scientific seeds are not added to fill slots.

Outputs: manifest.json, frozen-source/, source.patch, experiment-ledger.jsonl,
coordinator-status.json, train/*/progress.json, atomic checkpoints, all-episode
learning-curves.csv, dashboard.html, evaluation/*/summary.json, comparison.json,
report.md and exit-audit.json. Failed drain outcomes remain in all denominators;
success-only J cannot rank differing success subsets. Resource deadlock is not
inferred from any timeout. This local baseline records failures but does not
reconstruct the remote strict-closure/D3 diagnostic pipeline yet.
