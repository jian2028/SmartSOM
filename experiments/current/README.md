# Current RLlib campaign

This is the single maintained research campaign. Inputs are authored and pass
framework-free schema loading; this is not a formal launch or completed runtime
acceptance. See `campaign.json` for the complete inventory, including PR42 Morris/E1a-current PPO (ADR0101).

Small and Medium retain the V3 physical scenarios (300/750 orders), 32 distinct
3–6-operation route templates, conditions, and training seeds 101–105. Each map
has Base, implicit SL-attention and implicit SL-concat, shared/independent module
ownership, and official PPO/DQN recipes: 12 recipes per map. Algorithm parameters
start from pinned RLlib 2.60 defaults, with the approved thin same-version DQN
replay checkpoint extension (ADR0095). Replay and learner state share the official
checkpoint; old-version checkpoint compatibility is not provided. Concat
capacities and semantic action definitions are retained.

The training budget is 1000 observed completed episodes, checked at official
iteration boundaries; overshoot is recorded. Periodic evaluation milestones are
every 50 episodes. Both fixed panels contain 50 seeds per condition (200 worlds):
10001–10050 for periodic evaluation, 20001–20050 for final test, action seed 0.
These retain the V3 seed domains; hashes pin the generated worlds to this source.
The two algorithms share worlds, while PPO samples and DQN uses official
non-exploring evaluation. Test never selects a checkpoint or search objective.

All complete cases report
`J = makespan/5000 + mean_tardiness/300 + (1000/300)*(1-final_passing_rate)`.
Base/SL use the approved incremental team cost; no old V3 deadline penalty is
carried forward. Undiscounted team cost sums to J; official decision-step discounting
still applies during learning, so its discounted objective is not identical to J.
Drain expiry is truncation. Best minimizes full-panel mean J
only if every periodic case completes; last remains the official recovery
checkpoint. Both refer to official checkpoints. Historical scores/checkpoints
are not comparable continuations of this campaign and are not rewritten.

Drain values 10778 (Small) and 10590 (Medium) retain the Rule-derived fixed bounds
authored at `1e91c81`. This provenance is recorded per recipe in `campaign.json`;
it is not a claim that the new RLlib Rule adapter has been calibrated or verified.
The paper comparison retains one composite Rule, Base without social learning,
and implicit SL. Rule no longer expands into fixed slow/normal/fast recipes.
Random stays visible as an internal debugging reference, not a paper baseline.
Each map also has one `morris-ppo` recipe: PR42 centralized direct tasks, CONWIP,
64×64 candidate/value networks and the preserved overdue/failed-output reward
scaled by 0.001. Official PPO owns updates; fixed sampled panels select best by J.
Candidate rows stay compact until batch construction; medium-map learner memory
can be substantially larger than Base/SL, so resource profiles must reflect the
measured workload. Engineering validation is separate from a formal launch. Large stays
authored-only in the source archive. Do not substitute a learning policy for Rule.

The campaign Rule recipe is `configs/algorithms/current/rule.yaml`. It selects the earliest
due job, breaking ties by current-machine normal duration and semantic job ID.
Its critical ratio is `(due_at - now) / remaining_normal_ticks`: the denominator
contains the current operation on the deciding machine exactly once, followed by
the minimum normal duration on compatible machines for each remaining operation.
Each duration includes the machine rate and integer ceiling. CR < 1 selects fast;
1 <= CR < 1.2 selects normal; CR >= 1.2 selects slow. If the desired mode is not
legal, choose normal if legal, otherwise the shortest legal duration for that job.
Transport, queueing and future faults are not predicted. This heuristic assumes
the campaign's slow/normal/fast quality tradeoff; it does not minimize J directly.
Other reservation/dispatch rules and Random's legal-candidate sampling are unchanged.

Rule and Random fixed panels use `small/{rule,random}/experiment.yaml` and the
corresponding Medium recipes through the same `smartsom check/run` entry point.
They use the learning campaign's protocol permissions and direct legal semantic
actions; they do not sample learned rule aliases. `run.split` selects the fixed
`test` (default) or `periodic_evaluation` inventory. The new Rule changes actions
and scientific identity; inherited drain bounds above retain their old provenance,
not a claim of recalibration with this rule. Frozen runs and historical fixed-speed
recipes are preserved. See ADR0099 and ADR0100. Non-learning panels have no model
best/last. They reserve a Tune coordinator plus all Ray case workers; profile
`rllib.evaluation_num_env_runners` overrides experiment evaluation concurrency.
Evaluation CPU/memory declarations apply, without Learners or GPU workers.

Use `smartsom check` with a profile before a separately authorized launch:

```sh
uv run --no-sync smartsom check experiments/current/small/base-shared-ppo/experiment.yaml --profile configs/profiles/rllib-local.yaml
```

Profiles own driver/Learner/EnvRunner resources. Existing experiment concurrency
ceilings (Small 3, Medium 5) remain explicit; they do not guarantee admission of
that many trials. No resource calibration or online adjustment is introduced.

Debug/performance inputs must be generated from one of these recipes into a
temporary evidence directory, recording the parent hashes and exact overrides.
Do not maintain another benchmark campaign. Short debugging precedes medium
measurements. Compare the same actual episodes, environment/agent steps, learner
updates, evaluation and checkpoint work. Report initial startup separately,
episode duration/distribution, stable training, evaluation and checkpoint costs.
Per-episode timing alone is not equivalent work if truncation or update counts
differ. Only validated comparable results can satisfy the 5% performance gate.

Old authoring files are retained under `archive/legacy_runtime`; original V3
authoring is under `archive/v3_1e91c81`. Running server experiments remain bound
to their original checkout and frozen files.
