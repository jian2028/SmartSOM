# Rule-action DDQN on the narrow layouts

This is a configuration recipe, not an execution manifest or scientific result.
Launch only after the integrated review, CI, merge and experiment gates pass.

The four flat experiment entrypoints share eight embedded-geometry scenarios and
two native algorithms. `medium-sl` and `small-sl` use native ALL observations and
the attention encoder; `*-base` use Base. Both use the current rule-credit
runtime and Double DQN. The only standalone map catalog remains
`src/smartsom/config/maps/`: these scenarios embed `medium-factory` (27×12,
20 machines/20 AGVs) or `small-factory` (22×6, 8/8). No extra factory files are
introduced in `configs`.

Each experiment maps `condition_scenarios.normal/low/medium/high` to its four
scenario files. The filename `none` means normal; `moderate` means medium.
Normal explicitly disables breakdown. The other three files retain the same
complete historical interval dictionary; the condition selects its low, medium
or high profile. A filename does not select severity. Geometry and jobs are
identical across a map's four scenarios.

The inherited protocol is 320 orders, 40 initially available, mean interarrival
14.4 ticks, four historical route families, and the original due-date, processing
and fault distributions. Episodes end upon delivery of all orders or at last
release plus 2000 ticks. Each model trains 1000 episodes, balanced cyclically
across the four fault conditions (250 each), using world seeds
`seed_for(train_seed, "social-long-v1/world", "train", episode_index)` without a
modulus. Each YAML retains `train_seeds: [4101, 4102, 4103]`.

Both algorithms retain width 32, learning rate 0.0003, three gradient batches per
group, batch size 256, replay capacity 10000, target update interval 100, and
training exploration epsilon 0.1. Initialization is explicitly
`legacy-rule-dqn-v1`. The common physical reward is
`-(incremental overdue integral + 1000 * incremental unqualified deliveries) / 320`.
`learning_reward_scale: 0.001` matches the old driver's additional division
by 1000 before replay insertion. Undiscounted DDQN targets are implemented in the
learner; there is no invented YAML `gamma` field. Rule alias ordering is explicit;
STOP and EMPTY_WAIT are disabled.

There are no initial/intermediate validation evaluations or extra baselines.
Checkpoints are saved every 50 training episodes, but only episode 1000 receives
the 40-case test panel, with `argmax` (native DDQN greedy-Q) only. There is no
epsilon-greedy evaluation panel. `run.methods: [configured]` prevents default
baseline methods. `seconds: null`, `tune: off` and empty `case_seconds` avoid a
wall-clock scientific cutoff. `max_failures: 0` requires explicit failure retry.

Each entry explicitly sets `resources.guard` to an 8 GiB worker RSS ceiling,
8 GiB host-available floor and 2 GiB output-disk-free floor. This preserves the old
cooperative guard, separately from admission budgets: rollout checks are limited
to one per second, with forced model/update/checkpoint/receipt boundary checks.
A resource failure retains committed work and requires explicit recovery. It is
not an OS-enforced peak RSS limit; long update/serialization operations can cross
the threshold before their next check. The historical stage-two 80 GiB admission
threshold is unchanged. See [ADR0069](../../../../docs/decisions/0069-cooperative-worker-resource-guard.md).

`common/*-test.json` preserves the historical confirmation panel's exact 40 case
IDs, conditions, scenario seeds and action seeds, ten cases per fault condition.
The physical worlds and their hashes are regenerated for each new layout; they
are not claimed identical to the historical geometry. The compiler expands and
validates these recipes, and `case_set_sha256.test` pins the resulting inventory.
`common/provenance.json` records the historical source and bundled-map hashes.

From the repository root, static checks are:

```sh
uv run --no-sync smartsom check experiments/rule-dqn/narrow-layout --id medium-base --id medium-sl --first-train-seeds 1 --json
uv run --no-sync smartsom check experiments/rule-dqn/narrow-layout --resume-missing --json
```

After the gates pass, the authorized first launch is:

```sh
uv run --no-sync smartsom batch-run experiments/rule-dqn/narrow-layout --id medium-base --id medium-sl --first-train-seeds 1 --background
```

This freezes twelve task directories but activates only two models: medium Base
and ALL at seed 4101. Four remaining medium tasks and all six small tasks are
deferred. Preallocated directories are not evidence of a model starting. The
batch remains partial while deferred work exists. First-round final evaluation
contains 80 cases total. Small stage 1 does not automatically activate.

When the first invocation has exited and supplementation is authorized:

```sh
uv run --no-sync smartsom batch-run experiments/rule-dqn/narrow-layout --id medium-base --id medium-sl --resume-missing --background
```

It activates four missing medium tasks (seeds 4102/4103 per arm). Repeating that
same command after successful completion adds zero tasks. Pending final
evaluation continues through the official runtime; failed tasks require explicit
retry using their existing continuation, not an implicit restart. All calls use
the same bound output root `runs/rule-dqn/narrow-layout`, with artifacts under
`trials/<entry>/seed-<seed>`. Do not change scientific `--set` values between
invocations, and do not use `--set train_seeds=1` to select the authored first
seed: that existing scientific override means seed 101.

| Entry | Stage | Authored seeds | First launch | Later missing inventory |
| --- | --- | --- | --- | --- |
| medium-base | 0 | 4101, 4102, 4103 | 4101 | 4102, 4103 |
| medium-sl (ALL) | 0 | 4101, 4102, 4103 | 4101 | 4102, 4103 |
| small-base | 1 | 4101, 4102, 4103 | deferred | 4101, 4102, 4103 |
| small-sl (ALL) | 1 | 4101, 4102, 4103 | deferred | 4101, 4102, 4103 |

The complete optional inventory is twelve models / 480 final test cases.
Selection and live owner claims do not change the frozen scientific inventory.
Tracking is explicitly online in project `SmartSOM`; server credentials and the
final merged source revision remain launch-time acceptance requirements.
