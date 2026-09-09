# Resource-agent MARL acceptance

**Item 13 awaits formal macOS acceptance.** The first checkpoint, `467a31e`, adds
public resource projection, deterministic proposal coordination and the optional
PettingZoo API. Training/checkpoint/run/study integration has development passing
evidence after the authorized reward-unit adjustment. The earlier failed attempts
remain failed. Interface tests or parameter updates alone do not complete item 13;
a fresh clean-main training/evaluation report is required. Linux/h20 verification
is deferred to Week3 and is not implied by macOS completion.
See [ADR 0012](../decisions/0012-resource-agents-and-joint-proposals.md).

## IDETC field comparison

The frozen runtime is the **inner** `jobshoplab/jobshoplab/env/env_wrapper.py` at
729bc692c85781be275f42144ce582449b875d98. Its package shim prioritizes that directory.
Both verified byte hashes are in the committed source manifest. No external code
is needed for tests or execution.

| Frozen field/group | Current interpretation/difference |
| --- | --- |
| job_running, job_progression | Public processing/paused flag and completed operation count |
| job_executed_on_machine | Binary history reconstructed from completed semantic modes |
| machine_progression | Distinct completed jobs per machine, not operation count |
| available_jobs | Released, unfinished, not processing/paused; not a legality mask |
| current/global time | One public clock; remove duplicate feature |
| global input/output fill | Counts with configured scale; these areas are unlimited |
| local machine/buffer state | Current up/down, holding phase, membership and reservations |
| machine_remaining_time | Chosen nominal duration and elapsed wall simulation time; no internal expected completion |
| AGV remaining time | Known pickup/arrival only; unload waiting remains unknown |
| candidate_job/remaining_ops/duration | Per-resource full semantic choices, with explicit mode/destination and nominal/travel parameters |
| current_transition | Replaced by explicit information for every candidate |
| full hidden job directory/normalizers | Fixed declared capacities, reveal-bound slots and configured scales |
| truncated lists/modulo/fallback actions | Rejected; no truncation, remapping or conflict resampling |

## Independent checks

Crossing jobs start C1/M1 [0,2), D1/M2 [0,3) in the same joint round. The next
round starts C2/M2 [3,4), D2/M1 [3,5), makespan **5**. Two machines claiming A1
accept M1's action and reject M2 without choosing B1 as a fallback. Two zero-time
AGV proposals for A accept only the first, even though the second would become
a legal reroute after delivery. Processing beats a conflicting transport proposal.

All legal JA/MB/UPT/AGV/buffers/quality/holding switch combinations are checked,
including both arrival triggers and direct Transfer ownership. Fixed semantic
choices compare exact core trace, action replay and schedule/quality replay.
NOOP, public waiting, stalls, atomic rejection, reveal/hidden truth, fixed capacity,
initial advancement, limits and complete lifecycle are independently checked.

PettingZoo 1.27's official Parallel API test runs directly on the environment.
Its seed helper ignores masks, so a **test-only sampler** supplies the public mask
before drawing an action; env.step remains strict. That helper also tests only
one transition due to any(dict) on terminal dictionary keys. Independent seeded
full-episode comparisons supplement it; no external test source is modified.

```sh
uv sync --locked --extra pettingzoo
SMARTSOM_REQUIRE_PETTINGZOO=1 uv run --no-sync pytest -q \
  tests/unit/test_resource_projection.py tests/unit/test_pettingzoo.py
```

The real PPO gate is fixed at seed101, 4096 joint environment steps, two 64/64
role networks and the existing 4-job/9-operation micro. Evaluation uses study
seed202, replicas0–4, MARL+SPT (10 runs). No superiority requirement or claim of
equal sampling budgets with centralized PPO. Fixed-budget failure remains failure.
Actual Linux execution remains pending until run, regardless of CI configuration.

First-checkpoint verification on macOS: 305 focused resource/Parallel checks;
1488 full locked learning/CP tests; 1169 base tests and 19 explicit optional skips.
Ruff, formatting, dependency consistency, lock check and diff checks passed.
The first sandboxed full attempt failed because Ray could not bind localhost and
was interrupted; the completed unsandboxed local rerun is the passing evidence.
These are engineering checks, not the new MARL training or integrated acceptance.

## Second-checkpoint development outcome (2026-09-08)

The fixed recipe was unchanged: seed101, 4096 joint steps, 49152 agent steps,
16 PPO updates; two separate 64/64 actor networks, CPU, one numerical thread.
Every completed training attempt changed both actors and restored both exported
role modules without changing their parameter hashes. The final training ledger
audit reproduced all 167 episodes: 149 complete, 17 exploration failures and one
explicit budget-end prefix. No hidden environment input or physical rule changed.

| Attempt | Deterministic paired evaluation | Interpretation |
| --- | --- | --- |
| Initial development probe | MARL 5/5, SPT 5/5 | Preliminary success; not a substitute for later gate failure |
| First full gate, before batch-order correction | MARL 3/5, SPT 5/5 | Two all-NOOP policy stalls; full pytest failed |
| Gate after semantic batch-order correction | MARL 0/5, SPT 5/5 | Fixed-budget acceptance failed; second commit stopped |

The first two attempts had identical initial weights and first observations but
different first sampled joint indices. Installed Ray 2.58 source explains this:
ConnectorV2 iterates agent sets, AgentToModuleMapping preserves that iteration
order, and GetActions samples rows in that order. A controlled hash-seed 1/42
probe reproduced the association change. The adapter now sorts all columns by
episode sampling order/semantic resource ID before role batching. Independent
tests verify column alignment and identical resource/draw associations across
hash seeds. This fixes a framework protocol defect; it does not select a favorable
hash seed, change the root seed, mask, reward, PPO parameters, or input physics.

After the correction, all five MARL failures are `policy_stalled`, at ticks
86, 107, 91, 86, 86 for replications 0–4. Physical candidates still exist, but all
resources choose NOOP with no public future-event witness. The team return is
-10001; failed runs report no successful makespan or passing-rate metric.
No further training, budget increase, seed change or heuristic substitution was
performed. The next training design requires a separately agreed scope.

Local retained evidence (not committed):

- `artifacts/resource-marl/failed-semantic-order-gate/evaluation/report.json`
- `artifacts/resource-marl/failed-semantic-order-gate/failure_replay.json`
- `artifacts/resource-marl/failed-semantic-order-gate/pytest.log`
- `artifacts/resource-marl/failed-unsorted-framework-gate/`
- `artifacts/resource-marl/framework-order-diagnosis.json`

The copied attempts preserve their original absolute references as well as all
source/dependency/input identities, checkpoints, traces and records. They are
development evidence from a dirty working tree, not a formal new-main run.
The final full test gate stopped at 1 failed / 18 passed because the mandatory
10-run acceptance failed. Final base checks passed 1320 tests (24 optional skips).
The separate regression run excluding the two real-training test modules passed
1640 tests, including actual CP ft06=55, official FJSP=6 and Mk01=40. The new
RLlib protocol, column alignment and hash-seed row-association checks passed
separately (3 tests); Ruff, format, lock, dependency and diff checks passed.
The failed learning gate is never described as a passing full suite or an
adapter-only completion. Actual Ubuntu execution remains pending.

## Authorized uncommitted adjustment (2026-09-08)

The user subsequently requested diagnosis, adjustment and retraining without a
commit. This supersedes the prior stop on further development training, not the
failed attempt's recorded outcome or the requirement for integrated acceptance.

The retained checkpoint confirmed NOOP really had the highest per-action
probability in all five failed final states. The critic predictions stayed near
zero. Both role value losses were clamped to 10 on almost every update while the
unclipped squared errors were thousands to tens of millions. Installed Ray 2.58
source and an independent autograd probe verify that this clamp gives zero value
loss gradient when absolute error exceeds sqrt(10). Ray documents the parameter's
reward-scale sensitivity: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html

Before retraining, `artifacts/resource-marl/adjustment-20260908/protocol.md` fixed
one change: learner_reward_scale=0.0001, applied only to copied learner rewards
before GAE. The original case, root101, 4096 joint rounds, 16 updates, both 64/64
MLPs and all other PPO settings stay fixed. Core/Parallel/replay rewards remain
raw ticks; the failure cost remains 10001 and NOOP remains available. No fallback,
action filtering, favorable seed or intermediate checkpoint selection is used.
The first 256 sampled joint indices/rewards/ticks exactly match the old attempt.

The single adjustment attempt completed with 143 successful training episodes,
19 legitimate exploration failures and one budget-end prefix. Both actors changed
and saved/restored. Every episode's input, ledger, reward and trace audited; all
successful episodes also passed action and schedule replay. Value losses never
saturated across the 16 updates. Final explained variance was 0.665 for machines
and 0.725 for AGVs (these optimizer diagnostics are not performance claims).

| Replication | Deterministic resource PPO | SPT |
| --- | ---: | ---: |
| 0 | 233 | 120 |
| 1 | 225 | 135 |
| 2 | 205 | 147 |
| 3 | 205 | 135 |
| 4 | 205 | 136 |
| Mean makespan | 214.6 | 134.6 |
| Sample standard deviation | 13.4462 | 9.6073 |

All 10/10 paired runs completed and passed observation/action/schedule checks;
the five MARL runs also passed exact joint coordination replay. Raw team returns
remain exactly negative makespan. This meets the development completion/replay
gate, not superiority over SPT, broad policy reliability or a new main-commit gate.
The previous 0/5 result remains available and is not relabeled successful.

Evidence: `artifacts/resource-marl/adjustment-20260908/diagnosis.json`,
`trial-a-numerical-result.json`, `trial-a-evaluation/report.json`, with baseline
patches, current source hashes, logs, training and evaluation directories. The
training checkpoint is under `artifacts/resource-marl/training/20260908T111813407757Z-0e26339e931a45af97ad42ed0652b66c/`.
Checks: 161 focused tests; 1647 non-training regression tests, including actual CP
55/6/40; Ruff, formatting, lock and installed dependency checks. The training and
full audit command completed separately; this is not described as a full pytest
rerun including unchanged centralized backend training. No commit or push was
made. Actual Linux validation and new integrated-main acceptance remain pending.

## Engineering completion protocol (2026-09-09)

The fixed recipe remains seed101, 4096 joint rounds, 49152 agent steps and
16 updates, two 64/64 MLP roles and learner_reward_scale=0.0001. The case,
NOOP, masks, conflict protocol and physical rewards are unchanged. The script
checks frozen training/evaluation identities independently of mutable presets;
complete environment inputs and switches are included in paired-world identity.

`scripts/validate_resource_learning.py` defaults to formal mode. All training and
evaluation manifests and the auditor must identify one clean implementation
commit already integrated into local main. A detached worktree at that commit
is supported, including after a later documentation-only completion commit.
The actual imported source must belong to that checkout. Source checks run both
before validation and before a passing report is issued.

Precommit checks and the real-training integration test pass `--development`.
Their reports explicitly record `evidence_kind=development` and cannot populate
`accepted_source_sha`. Formal reports record the accepted implementation and
actual platform. Both modes require each of replications 0–4 to contain exactly
one resource PPO and one SPT evaluation, all ten physical/observation audits,
five joint replays, the training ledger audit, both role updates/restoration and
16 finite, unsaturated critic-loss records. Performance superiority is not a gate.

Focused coverage includes ordinary runner/Parallel/replay lifecycle equivalence
at initial advance, exact/overshot tick limits, round limits, stalled policy and
physical deadlock; writer failure after actual physical completion must preserve
trace and the original cause. Required MARL CI also runs the reward-scale and
acceptance-source regressions rather than skipping them for missing dependencies.

If fixed-recipe learning fails, retain and diagnose the failed evidence. Only
verified engineering defects may be fixed within this completion scope; changing
seeds, training budget, PPO settings or physical inputs requires a new decision.

### Fresh precommit checks

On macOS, the 2026-09-09 completion run passed 503 focused tests and the full
1691-test gate with all CP/Gym/PettingZoo/centralized-learning/MARL requirements
enabled (no skips; four existing framework warnings). Ruff, formatting, locked
dependency resolution and installed dependency consistency passed.

The full gate includes actual CP reference checks, centralized training and its
15/15 paired acceptance, plus one new resource-PPO training attempt and 10/10
paired development acceptance. The resource result remains MARL 214.6 versus
SPT 134.6 mean makespan. No seed, budget, action or physical-input changes were
made. These precommit results do not replace the fresh formal run below.

Retained evidence: `artifacts/resource-marl/engineering-20260909/full-gate.log`,
`focused.log` and `pytest-development/fixed-resource-ppo0/evaluation/report.json`
under that same directory, including all training/checkpoint and child-run bytes.
