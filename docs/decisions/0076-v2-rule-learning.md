# 0076 — V2 rule learning and matched manufacturing scenarios

Status: implementation in progress; no verification or experiment launch claimed.

The user authorized implementation, review, a pushed unmerged PR and experiments
from its reviewed fixed commit on 2026-11-08. This campaign-specific source rule
overrides the usual integrated-main/tag requirement. Existing experiments and V1
components remain preserved. The detailed source specification is the user's
V2 plan dated 2026-11-07 with 2026-11-08 updates and implementation attachment.

## Action and learning contracts

Supersedes ADR0063's mandatory alias merging for explicitly configured experiments:
rule aliases may be retained as distinct semantic candidates. The default continues
to merge them. An executed physical action must not count as selection of every
rule recommending it; recording identifies the selected rule separately.
Unavailable rules are excluded, not replaced by another rule. Rule identity does
not encode physical entity ordinals. Machine scores seven job rules jointly with
three processing modes. PRE, POST and Input each share a station encoder across
their two stages, with distinct stage candidate encoders and scoring heads.

Dispatcher pickup and delivery are separate learning subjects and parameter owners
in V2. Same-role entity sharing remains separately configurable. Transitions join
successive decisions of the same subject, including intervening physical rewards;
finishing a pickup is not an episode terminal. Base and ISL retain identical action,
reward, physical legality and learning contracts. ISL adds the plan's candidate-
conditioned attention, including the two-level job/downstream relation for source
acceptance. Learned encodings belong in networks, not observation callbacks.

## Scenario and workload

Small/Medium/Large use 8/20/40 machines, 10/25/50 AGVs and 300/750/1500 jobs.
PRE and POST capacity is four; machine and AGV job capacity is one. Eight operation
types and exactly two capabilities per machine give each operation a balanced
compatibility ratio of 25% (2/5/10 machines). Operation pairs must also be balanced.
Retain two-cell interior aisles, one-cell upper/lower roads and scale I/O service.

The existing study's job length is four operations with an independent 0.5
probability of repeating the first operation immediately: lengths four and five
each have probability 0.5. Expanding the operation catalog must preserve that
distribution, route-family selection and processing perturbation mechanism.
Old per-operation base work is 30/40/50/40 with uniform relative variation ±0.2.
New operation labels need an explicit mapping preserving that base-work mix.
Arrival intervals remain exponential; any scale-dependent parameter change is
recorded against the previous scenario before experiments.

Preserve the marginal due multiplier distribution: 0.8 Uniform(2,3) plus
0.2 Uniform(1.5,2.5), multiplying the existing nominal processing/transport
baseline. Remove independent rush classification and priority semantics rather
than dropping the tighter component. Fault conditions remain normal/low/medium/high;
any engineering calibration changes record old/new parameters and physical reasons,
never method rankings. Compatible next operations continue on the same machine,
without transport or queue insertion, selecting a new mode each operation.

## Comparison and acceptance

Implement V2 Base/ISL before the Morris adaptation. Verify the remote Morris report
and associated source SHA. All comparators use this simulator's Mover; retain Justin
rules. Name partial Morris reproduction as E1a-current adaptation, with differences.
Keep Double DQN for Base/ISL. Reward must avoid rewarding slow or unproductive busy
time; document its relationship to tardiness and utilization after source review.

Training seeds are 101,102,103, in that order within each scale; methods may run
concurrently within a seed. Preserve the episode budget and four-condition mixing.
Each checkpoint evaluation has 20 fixed cases per condition (80 total), retaining
the prior ten seed identities and deterministically extending them; no increase in
checkpoint frequency. New physical scenes require new world hashes, not reused old
hashes. Separate training, environment and policy seeds. Justin has no training seeds.

Before publication: focused tests, appropriate repository checks, at least three
serial alternating uninstrumented performance comparisons (rollout and update),
and independent engineering, research and test reviews. Investigate regressions
above 5%, document justified functional costs. Freeze reviewed code and inputs.
Start Small on Mac and Medium on BudgetVM only. Verify actual sampling, updates,
checkpoints, evaluation and Justin progress, with a durable later-seed queue.
Large is engineered and checked but no formal training starts. Final evidence
includes PR/SHA, paths, adaptation decisions, tests/reviews/performance, run status
and official monitor/stop/resume/cleanup commands.

### Engineering calibration, 2026-11-08

The initial compact generated scene accidentally left only one cell of road along
both I/O sides. A 1200-tick Small seed-19 probe had 16 Base / 17 Justin decisions,
with opposing empty vehicles stalled on the input-side road. The corrected layout
uses two-cell side aisles as well as two-cell internal aisles; only the top/bottom
edge roads remain one cell wide. Machine coordinates shift one cell right, with
width `5 * machine_columns + 4`. The same fixed-tick probe then produced 492 Base,
1569 Justin, 942 ISL and 1047 Morris decisions (completed jobs 0/35/19/2).
These are untrained engineering probes, not algorithm-performance evidence or a
basis for changing breakdown parameters. All scale panels are regenerated from
the corrected geometry, preserving their seed inventories.

The official batch requires uniform execution settings across child files. The
initial recipe therefore uses three concurrent tasks with three evaluation workers
each (at most nine case workers), pending throughput calibration against the target
of ten. It does not allocate ten workers independently to each method.

Independent stage freezing is configured through `frozen_stages`, e.g.
`[dispatcher.pickup]`, only for stages already separated by `split_stages`.
Freezing prevents parameter/optimizer/target updates; it does not alter physical
decisions or exploration. DQN reports cumulative observed and omitted transition
counts and omitted fractions per physical learning subject, including stages with
100% omitted tails and no learner update; these counts survive checkpoint restore.

Exogenous capacity calibration with seed777 showed the inherited2000-tick drain
made completion physically impossible in faulted scenes even at fast mode with
zero transport loss (Large high: availability.699, fast work/capacity1.354). Retain
fault distributions; set the common training/evaluation drain to10000 ticks.
Expected Small slow-mode work64800/(8×.7)≈11571 total ticks; last arrival≈3740
plus10000 gives capacity margin. All scales have proportional load. This fixes a
horizon constraint, not any method's ranking; incomplete metrics remain censored.
