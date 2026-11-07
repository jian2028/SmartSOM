# V2 matched experiment campaign

Status: engineering verification; formal launch receipts are recorded separately.
This directory contains 10 official recipes per scale: Base, ISL and Morris
training seeds101/102/103, plus one Justin evaluation. Invoke a scale directory
with `smartsom check` / `smartsom run`; stages1/2/3 enforce whole-group seed order.
Do not launch this parent directory: Large is validation-only in this campaign.

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
checkpoint save every50 episodes, final-only evaluation at1000 (no added evaluation
frequency). Stage1 includes seed101 and Justin; stage2 seed102; stage3 seed103.
Justin has no training models. Current conservative execution is three tasks ×
three evaluation workers (at most nine case workers); tune throughput against the
target10 without creating independent ten-worker pools per method.

Formal deployment must use the same reviewed fixed commit on Mac Small and
BudgetVM Medium, record frozen source/config/component identities and durable
queues, and leave the PR unmerged. Large receives no formal launch. Run receipts,
exact IDs and stop/resume commands belong in the final campaign handoff.
