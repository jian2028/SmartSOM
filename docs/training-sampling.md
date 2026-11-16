# Official RLlib training configuration

`smartsom check/run` compiles the scientific inputs and launch profile. New learning
runs use official RLlib PPO/DQN with Tune FIFO; there is no native training fallback.
See [ADR0098](decisions/0098-rllib-core-closeout.md) for current acceptance scope.

## Sampling and resources

Set component counts and logical resources in a profile:

```yaml
resources:
  cpus: 8
rllib:
  resources:
    num_cpus_for_main_process: 1
  learners:
    num_learners: 0
  env_runners:
    num_env_runners: 2
    num_cpus_per_env_runner: 1
  evaluation_num_env_runners: 2
concurrency:
  max_concurrent: 1
```

The Learner is local when `num_learners: 0`; the driver owns its resource request.
MARL uses one factory per EnvRunner, with semantic agent-to-module mapping for
shared or independent parameters. Different agents retain distinct trajectories.
RLlib owns sample collection and updates; there is no SmartSOM synchronous wave,
physical-time discount or update-credit scheduler. Omitted settings use the pinned
framework defaults, recorded in the resolved configuration.

Base/SL sampling is event-driven: a production decision opens one learned WAIT
at the end of its negotiation interval. Continuing to wait does not generate a
new agent step at every physical tick. Physics and team costs still advance over
every interval, and final observations retain valid bootstrap actions for all
started agents. This reduces trajectory storage but changes official decision-step
discounting and sample counts; see [ADR0102](decisions/0102-event-driven-learning-waits.md).

Training and evaluation are sequential; evaluation cases run concurrently on
resident evaluation EnvRunners. All resident components, including idle workers,
count toward the Tune request. Resource declarations control admission, not OS
CPU or memory isolation. Worker counts and concurrency caps are set at launch.
No online resizing or custom throughput calibration is performed.

## Algorithm parameters

Start from `configs/algorithms/rllib-base-ppo.yaml` or `rllib-base-dqn.yaml`, or the
maintained `configs/algorithms/current` research recipes. `learner.params` contains
public `PPOConfig.training()` / `DQNConfig.training()` keyword arguments for the
locked version. For example, a small PPO engineering check may explicitly set:

```yaml
learner:
  target: smartsom.learning.rllib_config.ppo_config
  params:
    lr: 0.0003
    train_batch_size_per_learner: 64
    minibatch_size: 32
    num_epochs: 1
```

This small batch is a correctness fixture, not a research recommendation. DQN
uses the official loss, target updates, exploration schedule and replay; its
supported research heads require `num_atoms: 1` and `noisy: false`. The approved
replay state extension keeps replay inside the official checkpoint. Do not substitute
an unrelated replay class or disable replay persistence. The artificial zero-Q
exploration boundary remains documented in ADR0089; no custom epsilon-greedy is added.

Retired native fields such as `epochs`, `batch_size`, `learning_rate`,
`update_trigger`, update credits and target clocks are rejected. Use their official
configuration equivalents explicitly rather than assuming identical semantics.
The CLI convenience `--learning-rate` maps to official `lr` for the new factories.
Nondefault native `sampling` and phase `trial_resources` settings also produce
migration errors; use the resident component profile above.

## Evaluation and recovery

PPO evaluation uses policy sampling and DQN uses its default non-exploring action
selection. Best minimizes mean J over a complete fixed periodic panel under that
same selection policy. Incomplete panels cannot select best. Final evaluation
loads the selected official best/last checkpoints; the trial completes afterward.

Pause, budgets and checkpoint commits use official iteration boundaries. Actual
physical ticks/episodes can exceed the configured stopping threshold, and those
counts are recorded. Whole-run resume uses `Tuner.restore`; it preserves the
original finite deadline and completed results. Recovery retains official model,
optimizer, counters and DQN replay state, without promising mid-simulation or
bitwise RNG continuation. Native model files are not resumed across frameworks.

Historical controls and exact-wave tests are preserved in ADR0082/0083 and the
[test migration archive](../archive/rllib-transition-tests/README.md). They do not
specify the new framework. Representative performance evidence is reported
separately from correctness tests and scientific conclusions.
