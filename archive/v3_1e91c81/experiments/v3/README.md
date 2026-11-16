# V3 experiment preparation

This is the single maintained V3 campaign, prepared on PR41's integrated main
(`f29eff0`). It is a new scientific configuration, not a continuation of an old
checkpoint. Small/Medium deadline calibration and engineering checks are complete; formal
training has not started. Historical sources and results remain unchanged.

## Confirmed workloads

| Map | Machines | Types per machine | Candidates per type | Orders | Initial | Mean arrival interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 8 | 2 | 2 | 300 | 40 | 14.4 ticks |
| Medium | 20 | 2 | 5 | 750 | 100 | 5.76 ticks |
| Large, authored only | 40 | 2 | 10 | 1500 | 200 | 2.88 ticks |

Preserve existing machine positions, capability pairs, AGVs, roads, buffers,
breakdowns, quality modes and due-date generation. Capability counts are balanced;
this does not claim identical physical loads or measured utilization per machine.
Arrivals remain exponential interarrival draws accumulated before integer-tick
release rounding. Initial and total order counts are fixed per map.

Use 32 fixed route templates: eight each of lengths 3, 4, 5 and 6. Every job has
distinct types from A-H; no adjacent or nonadjacent type repetitions. In each
length group every positional column contains A-H exactly once, so each type
appears K times among the eight length-K routes. The full template list is visible
in the Scenario YAML and `scripts/build_v3_experiments.py`.

The existing generator uniformly samples the 32 templates. With exactly eight
templates per length this gives the approved joint distribution: each length has
probability 1/4 and each template conditional on length has probability 1/8.
It does not force exact counts in an individual episode. Sampling directly from
the fixed library avoids an unnecessary new generator API or extra RNG draw.

Base work remains A/E=30, C/G=50 and B/D/F/H=40. Each operation draws a continuous
uniform multiplier in [0.8,1.2] once at instance generation. Apply the processing
mode multiplier to the raw reference work and finally ceil to integer ticks.
No additional execution-time noise or repeat operation is introduced.

## Confirmed DQN learning settings

Base/SL each have shared and independent variants, seeds 101/102/103/104/105, 1000 training
episodes, and the existing balanced cycle of four breakdown conditions. Both
variants use the same settings below; network ownership is the experimental factor.

| Parameter | Value |
| --- | --- |
| update_trigger / update_every | transitions / 128 |
| update_amount / updates_per_transition | ratio / 0.0078125 (=1/128) |
| batch_size / warmup_transitions | 256 / 1024 per network |
| target_update_clock / target_update_interval | network_gradient_steps / 100 |
| replay_capacity / learning_rate | 10000 / 0.0003 |
| gamma / learning_reward_scale | 1 / 1 |
| exploration / epsilon / epsilon_final | linear / 0.30 / 0.05 |
| sampling.num_envs / processes / fragment_ticks | 2 / 2 / 256 |

Nominal replay intensity is two sampled rows per new inserted transition after
warmup; sampling remains random, and measured cumulative intensity includes warmup.
`epochs` is inactive for ratio updates. Thresholds are checked at collection
boundaries. Fragment ticks are neither a transition count nor a task deadline.
The two-environment setting is a throughput-validation candidate, not a measured
speed optimum. Per-task workers and batch/evaluation concurrency are separate.

Explicit epsilon decay transition counts are based on old per-network insertion
rates, aiming at approximately 200 episodes, and require verification with the
new workload. They are initial values, not calibrated optimal hyperparameters.

| Network | Small | Medium |
| --- | ---: | ---: |
| Shared dispatcher (pickup and delivery) | 300000 | 900000 |
| Shared machine | 260000 | 680000 |
| Shared pre/post | 210000 | 600000 |
| Input, either ownership | 60000 | 150000 |
| Other independent entity networks | 30000 | 32000 |

Large shared/input decay counts are provisionally twice Medium; other independent
counts are 32000. Large is not authorized for launch in this campaign.
Rule/Random retain their existing controllers. Central retains PPO episode updates
and serial sampling, and is authored only. DQN-specific schedules do not apply to
PPO. There is no extra 1/2/4-ratio campaign in the launch inventory.

## Implicit SL variants

`base-{shared,independent}` remains the original Base. Existing
`isl-{shared,independent}` is **implicit SL-attention**;
`isl-concat-{shared,independent}` is **implicit SL-concat**. Both variants are
authored for Small/Medium/Large, with unchanged scenarios, rewards, DQN settings,
seeds, validation/test panels and ownership. Large remains authored only.
No existing run is upgraded or new campaign launched by these files.

Both the observation and network component declare identical `params.social`:

```yaml
# Base
social:
  kind: none
# SL-attention
social:
  kind: implicit
  fusion: attention
# SL-concat, Small
social:
  kind: implicit
  fusion: concat
  slots:
    stations: 2
    vehicles: 10
    input_jobs: 300
    post_jobs: 5
```

The three blocks are alternatives, not repeated keys in one YAML mapping.
`kind: explicit` is reserved and raises an unsupported-mode error. It does not
silently behave like implicit SL. Legacy `social_enabled` remains accepted for
historical recipes; do not combine it with `social`.

Concat uses the same candidate-linked information as attention, with raw feature
vectors sorted by semantic ID within each relation, zero padding and mask bits.
It preserves each source job's downstream relation. Slots are ordered ranks, not
persistent entity IDs. There are no learned social encoders or attention layers.
Overflow raises an error instead of dropping peers/jobs. Bounds must cover the
scenario: stations means maximum compatible stations for one operation; Input
includes all possible original jobs; POST includes its capacity plus one job on
the machine. Medium bounds are 5/25/750/5; Large bounds are 10/50/1500/5.

Base is unchanged and **not parameter matched** to either SL model. Total online
Q-network parameters for the six shared subjects (excluding target copies):

| Map | Base | SL-attention | SL-concat |
| --- | ---: | ---: | ---: |
| Small | 75,113 | 167,913 | 596,521 |
| Medium | 75,113 | 167,913 | 2,814,121 |
| Large | 75,113 | 167,913 | 10,350,121 |

Most concat growth comes from retaining every Input job/downstream relation.
Fewer modules therefore does not mean fewer parameters, less memory, faster
training or better learning. Independent ownership makes separate entity copies;
its total parameter count is not the six-subject total above.

Use the existing official entry points, e.g.:

```sh
uv run smartsom check experiments/v3/small/isl-concat-shared/experiment.yaml
# After choosing a fixed source and authorizing a new run:
uv run smartsom run experiments/v3/small/isl-concat-shared/experiment.yaml --select 'train_seeds: [101]'
```

Keep ongoing experiments bound to their original clean source and frozen inputs.
Historical checkpoints remain usable there; this new source/configuration has a
different scientific identity and is not an in-place continuation of those runs.
See [ADR0086](../../docs/decisions/0086-implicit-social-fusion.md).

## Preserved V3 scientific definitions

Learned pickup selects NPF, HPOF, EDSF or LPCL. HPOF ranks POST occupancy ratio,
then resident PRE plus machine job count, then semantic identity. Keep existing
rule feature IDs. Base and implicit attention retain their network architecture;
the additional implicit concat variant is described below. Rule and Central retain
their respective action definitions.

Team reward is `-(dt + overdue_delta/N + 1000*failed_delta/N)/1000`, with fixed
original demand count N. At a real drain deadline with unfinished demands, charge
`10 + 10*unfinished/N` once, outside the /1000 scale. There is no completion bonus.
Idle time/utilization are diagnostics. At the task deadline DQN retains final
decision returns without bootstrap; PPO uses zero boundary value. Ordinary
fragments and wall/budget interruptions must not receive this task-deadline penalty.

Validation runs every 50 episodes with 50 seeds per condition (200 worlds);
test also uses 50 per condition in a disjoint seed domain. Both use argmax and
request up to 20 evaluation processes, subject to host admission. Choose best
by completed-world count, then mean delivery fraction, then mean observed cumulative cost including deadline penalties, then earlier episode (`argmax_delivery_cost`). The automatic final
test evaluates last; test best separately with the same recipe/panel and the
current `smartsom run --task evaluate --checkpoint /absolute/path/best.pt`
interface. Report both on the same test panel. Test never selects a model or
training parameter.
Keep best/last recovery-safe retention and W&B online in SmartSOM.

## Required calibration before training

Small and Medium were calibrated on 2026-11-11 using Rule normal on seeds
71001..71008 under all four conditions (32 worlds per map), disjoint from both
validation and test. Both panels completed every world within the initial
10000-tick drain observation window. Each map uses `ceil(1.5 * max tail)`,
where tail = last delivery minus last arrival. All learned/rule/random methods
on a map use the same explicitly authored value; no runtime result lookup.

| Map | Complete calibration worlds | Maximum tail | Authored drain_ticks |
| --- | ---: | ---: | ---: |
| small | 32/32 | 7185 | 10778 |
| medium | 32/32 | 7060 | 10590 |

Development calibration source was the current PR41-based V3 implementation;
these runs are setup evidence, not algorithm-comparison results. The observed
maximum is not a guarantee that every future case completes.

- small: [W&B calibration](https://wandb.ai/jianni-usc/SmartSOM/runs/e0ad2284890a4d9fb480ade422e68313); case receipt SHA256 `d48adeba99843a5f0a624bb359af8c87643ed01c93cf3fec43ff902e6308faed`.
- medium: [W&B calibration](https://wandb.ai/jianni-usc/SmartSOM/runs/1b698bb1d4d84f94829fa7927b08be34); case receipt SHA256 `9740caf7ccc8b40ccfcb567c7ba6d7831f84964fb769e7efcde8709e62461873`.

Large remains authored-only with provisional drain_ticks=10000; it is not selected.
Dedicated calibration recipes retain the original observation window. The builder
preserves already-authored campaign deadlines.

## Entry points and readiness

Use the existing `smartsom check`, `run`, `status`, `configure`, `pause`, `resume`
and evaluation interfaces. Run only Small on Mac and Medium on the server after
capacity qualification. Never launch the whole V3 root indiscriminately: it also
contains calibration recipes and authored-only Central/Large definitions.

Preparation requires configuration parsing, workload invariants, task-deadline
versus fragment tests, two-environment recovery tests, and applicable CI checks.
Local verification completed 1675 tests with two platform skips, plus an official
CLI best/last retention smoke. Paired unchanged-Rule checks preserved semantic
results; the final Small three-pair median timing difference was +0.52%. These
checks do not substitute for independent review or the complete hosted CI matrix,
which remain outstanding. Formal launch still requires an integrated main commit
or an explicit source tag.

## Current launch selection (2026-11-11)

Each learned recipe authors five training seeds, 101–105. Native execution-only
`--first-train-seeds 1` selects seed101 without changing the frozen seed inventory.
Mac Small selects Base-shared/SL-shared; server Medium selects Base/SL shared and
independent. Both also select Rule slow/normal/fast and Random, each on the complete
200-world test panel. All use W&B online. Central retains its PPO algorithm and
five-seed definition but is not selected; Large is also not selected.
Rule speed is an existing policy component's `processing_mode` parameter; all
three share the same physical scenario, reward, seeds and deadline.

These choices supersede ADR0084's original panel counts; see ADR0085.

Initial native admission budgets are Small: three tasks, 12 CPUs and 32 GiB;
Medium: five tasks, 24 CPUs and 80 GiB. This permits the selected training tasks
plus one baseline test task, with the remaining tests queued. Both request up to
20 evaluation processes; aggregate CPU and live memory admission may admit fewer.
Worker RSS guards are 16 GiB (Small) and 32 GiB (Medium). These are operational
limits, not allocations or OS isolation. Full-state evaluation loading is still
used; inference-only loading is not part of this launch change.

After source freezing and calibration, the current official launch selections are:

```sh
uv run --no-sync smartsom run experiments/v3/small \
  --id base-shared --id isl-shared \
  --id rule-normal --id rule-fast --id rule-slow --id random \
  --first-train-seeds 1 --background --wandb online \
  --output-root /Users/jianni/code/SmartSOM/runs/v3-20261111

uv run --no-sync smartsom run experiments/v3/medium \
  --id base-shared --id isl-shared --id base-independent --id isl-independent \
  --id rule-normal --id rule-fast --id rule-slow --id random \
  --first-train-seeds 1 --background --wandb online \
  --output-root /home/jianni/runs/v3-20261111
```

These commands freeze the full authored seed inventory and select only the named
tasks/first training seed. Do not replace them with an unrestricted directory run.
