# V2 matched campaign: shared, independent and comparators

Status: engineering verification; formal launch receipts are recorded separately.
Each map lives under `experiments/v2/{small,medium,large}`. Every map has
17 recipes: Base/ISL shared and independent seeds 101/102/103, Morris seeds
101/102/103, one Justin evaluation and one Random evaluation. Each recipe is
`<method>-<seed>/experiment.yaml`; results from a fresh map batch share the same
map output directory. Algorithm definitions remain under `configs/algorithms`.

```text
v2/<map>/
  base-shared-{101,102,103}/
  isl-shared-{101,102,103}/
  base-independent-{101,102,103}/
  isl-independent-{101,102,103}/
  morris-{101,102,103}/
  justin-101/
  random-101/
```

Stages 1/2/3 retain seed order. Select one map, not the whole `v2` directory.
Mac uses Small and BudgetVM uses Medium. Large remains validation-only.
Existing runs retain frozen source, inputs and outputs; this reorganization does
not migrate or restart their queues. The stage-0 server additions under
`runs/v2-expanded-20261108` remain a separate existing operational queue.

## Run only Morris on Medium

From a Linux/macOS checkout with the locked environment installed:

```sh
uv sync --locked --extra learning --extra cpu --extra wandb
source .venv/bin/activate
smartsom check experiments/v2/medium --id morris-101 --id morris-102 --id morris-103 --evaluation-processes 2 --wandb offline
smartsom run experiments/v2/medium --id morris-101 --id morris-102 --id morris-103 --evaluation-processes 2 --wandb offline --output-root runs/morris-medium --foreground
# After a normal stop, continue the same run:
smartsom resume runs/morris-medium --foreground
```

These commands select only Morris and execute the three seed stages in order.
Windows PCs use Ubuntu/WSL2 and its Linux Python environment, as described in
[CI guidance](../../docs/ci.md#experiments-on-windows-hardware).
Formal runs still require integrated main or an explicit source tag.

## Independent ownership and Random

Independent Base/ISL change only parameter ownership: each physical Machine,
PRE, POST and Input owns its model, optimizer, target and replay. Dispatcher
pickup/delivery retain separate subjects per AGV. Accept/service heads within
one buffer share that buffer's network. Small/Medium/Large have 45/111/221 owners.
Same-role entities start with identical parameter values in separate storage.
Equal episode budgets do not mean equal update counts or aggregate replay size.

`v2-base-random.yaml` uses the Base reservation mechanism, the exact V2 Base
rule-action catalog and permissions, and uniformly samples the presented legal
rule-action candidates. Rule aliases are retained, exactly as in Base; this is
not uniform sampling over deduplicated physical actions. It has no learner or
training checkpoint. `random-101` evaluates the same 80 pinned worlds and action
seeds as the other methods. Its `argmax` evaluation label does not make the random
policy deterministic: the policy samples using its seeded RNG.

Every method, including Random, uses the same
`smartsom.traffic.rule_mover.create_mover` with the same parameters. Randomness
applies to production decisions only; the mover is not randomized.

## Components and source

- `configs/algorithms/v2-{base,isl}-ddqn.yaml`: retained rule identities, seven
  Machine rules × three modes, separate Dispatcher pickup/delivery subjects.
- `src/smartsom/representations/v2/`: public derived facts, rules, observation and
  Torch networks. Base and ISL share all physical, action, reward and learning
  contracts; ISL adds candidate-conditioned station/job attention.
- `configs/algorithms/v2-morris-e1a-current-ppo.yaml`: explicitly an E1a-current
  adaptation, not a reproduction of remote experiments.
- `configs/algorithms/justin.yaml`: existing Justin rules, unchanged.
- Every method uses `smartsom.traffic.rule_mover.create_mover`.
- `scripts/build_v2_scenarios.py` and `scripts/build_v2_experiments.py` deterministically
  author scenes and pinned test panels. Regeneration does not start experiments.

Morris source reviewed: branch `Morris-11012026`, commit
`3b957785e572d002a608468430e195f64141bc84`, report
`reports/2026-11-05/notes__e1a_summary/e1a_summary.md`, algorithm
`configs/algorithms/central01_v2_e1a.yaml`, and corresponding
`learning/production_models.py`, `production_ray.py`, `experiments/task_reward.py`.
The native adaptation retains PPO1e-4, entropy.03, clip.2, four epochs,
minibatch100, gamma1, GAE.95 and 64×64 Tanh networks. Remote prefix-GRU dispatch,
feature layout, RLlib implementation and 256-tick update windows are not copied;
our adaptation uses current direct central actions, current public observations,
episode updates, common scenarios/reward and local Mover. Central action authority
also differs from the role controllers. This is not a control-structure-only ablation.

## Physical scenarios

| Scale | Machines | AGVs | Jobs | Compatible machines/type | Grid |
|---|---:|---:|---:|---:|---|
| Small |8|10|300|2 (25%)|14×12|
| Medium |20|25|750|5 (25%)|24×15|
| Large |40|50|1500|10 (25%)|29×24|

Each machine supports exactly two of eight types, using balanced round-robin
pairings; pair multiplicities differ by at most one. PRE/POST capacities4,
machine/AGV capacities1. I/O pools have2/5/10 interaction ports. Side and internal
main aisles are two cells wide; top/bottom edge roads are one. The initial compact
one-cell side-aisle prototype failed engineering traffic checks and was replaced
before freezing panels (ADR0076). The prior40-machine template was inspected;
its inspection/scrap layout is not the current final-Output contract, so the same
station-unit style is repacked into a near-square machine region.

**Job length remains4 or5, each with probability.5**: four base operations and an
independent optional immediate repeat of the first. Eight types do not mean eight
operations per job. Eight cyclic route families extend the catalog; base work
30/40/50/40 repeats over the eight labels, with uniform±20% perturbation. The next
operation stays on the same machine when compatible and chooses a fresh mode.

Arrival intervals remain exponential. Small retains14.4 ticks; Medium/Large use
5.76/2.88 to retain the Small offered workload per machine. The previous Medium
recipe also used14.4; this scale adjustment is explicit. Initial orders scale
40/100/200 (five per machine). Due multipliers retain the previous marginal
mixture `.2 Uniform(1.5,2.5) + .8 Uniform(2,3)`, multiplied by the nominal
processing/transport baseline. No rush label or separate priority remains.
Physical due dates change with the new scene and capabilities.

Fault parameters are retained after exogenous physical calibration: normal has no
breakdowns; low/medium/high intervals are truncated exponential with scales
600/300/150 and bounds[60,1800]/[30,900]/[15,450]. Repair is scale60,
bounds[10,180]. Methods share conditions and matching world seeds; calibration
must not tune these to a method ranking.

The old2000-tick drain failed an algorithm-independent capacity check: seed777
Large high faults had availability.699 and required fastest-mode work / available
capacity1.354, so even ideal routing could not finish. The common drain is now10000
ticks after last arrival for both training and evaluation. Expected300-job slow-mode
work64800 divided by8 machines×.7 availability needs about11571 total ticks;
last arrival near3740 plus10000 allows capacity margin, without assuming a method
wins. Medium/Large scale jobs and machines proportionally. This is a safety horizon,
not a completion guarantee under congestion. Final metrics remain censored if
jobs are unfinished. The80-world hashes include this changed horizon.

## Objective and metric meanings

All learning methods use `-(overdue integral + 1000 × failed inspections)/orders`,
with gamma1 and learning scale.001. This is the inherited common tardiness/quality
objective. It does not reward busy time or slow processing. Machine utilization
is a secondary physical evaluation, not a proven jointly optimized reward target.
Gui et al. (2024), [DOI10.1016/j.aei.2024.102646](https://doi.org/10.1016/j.aei.2024.102646),
§4.4 supports a common tardiness-difference reward. Zhang et al. (2024),
[DOI10.1016/j.aei.2024.102872](https://doi.org/10.1016/j.aei.2024.102872),
§4.1.3 motivates direct tardiness objectives rather than inaccurate proxies.
Neither establishes1000 as an optimal quality coefficient. Relevant Zotero full-text
passages and remote code were inspected in the independent research review.

`mean_tardiness` is all-order mean positive shipment lateness and is reported only
when all jobs finish; unfinished cases retain completion counts and observed overdue
integral, without treating unshipped jobs as on time. Passing rate over completed
jobs is distinct from final all-order passing rate. Aggregates censor final metrics
when cases are incomplete. Machine state ticks partition processing/idle/blocked/down;
down takes precedence. Processing fraction divides by machine-count×elapsed-ticks;
available-processing fraction removes down ticks from the denominator. Blocked and
idle time are not productive processing. Reports also retain flow/late tails,
traffic and buffer metrics, breakdown events and physical versus training reward.

## Panels and schedule

Each scale has80 pinned worlds:20 each normal/low/medium/high. Small/Medium retain
the first10 previous environment/action seeds per condition; the additional10 use
SHA256 namespace `smartsom-v2-panel/v1/{scale}/{condition}/{index}/{purpose}`.
Large receives its own deterministic panel. World hashes are regenerated against
new physics, never borrowed from old scenes. Every method and training seed within
a scale has the identical panel. Training, environment and policy seeds are distinct.

Learning budgets remain1000 episodes, balanced four-condition cycling,
validation every50 episodes (50 through1000). Each validation uses12 fixed worlds:
three each normal/low/medium/high, separate from the80 final test worlds. All
learning methods and training seeds share this validation panel within a scale.
Validation uses argmax and selects fewer incomplete cases first, then lower mean
joint cost when every case completes, then the earlier episode for ties.

Learning recipes use `checkpoint_retention: best_last`. `last.pt` is updated after
every committed episode with complete continuation state; `best.pt` is selected by
validation and first appears after episode50. Only the current owned recovery
generation remains; periodic and completed validation model copies are removed.
Validation scores, case receipts and model provenance remain. Atomic publication
may temporarily require both old and new models; hard links share model storage
where supported, with a copying fallback. This policy does not migrate existing
frozen runs or delete their old checkpoints.

The prescribed final test still evaluates the episode1000 last model on80 worlds;
best is retained separately and test scores never select it. Stage1 includes
seed101, Justin and Random; stage2 seed102; stage3 seed103.
Justin and Random have no training models. Recipes default to three tasks ×
three evaluation workers (at most nine case workers). Operational overrides may
change these limits; tune them against the target host and retain resource guards.

Formal deployment must use the same reviewed fixed commit on Mac Small and
BudgetVM Medium, record frozen source/config/component identities and durable
queues, and leave the PR unmerged. Large receives no formal launch. Run receipts,
exact IDs and stop/resume commands belong in the final campaign handoff.

### Morris memory representation

The E1a actor-v2 public payload stores three semantic owner associations per
candidate instead of three dense one-hot vectors. The first affine layer computes
the global term once, adds numeric features and gathers the corresponding one-hot
weight columns. Network dimensions, legal candidates and mathematical scorer are
unchanged. Floating-point addition order changes, so bitwise identical learned
parameters are not claimed. Dense/compact logits and gradients have independent
numerical-equivalence tests. A three-repeat serial Medium60-tick probe (up to12501
candidates) reduced median rollout9.43→4.70s, update29.25→3.60s and peak RSS28.19→2.51GB;
actions/rewards matched. These are engineering measurements, not full-episode
training throughput or algorithm-quality results.
