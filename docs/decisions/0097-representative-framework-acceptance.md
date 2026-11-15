# 0097 — Representative framework acceptance without trajectory alignment

The user narrowed this migration's performance scope to one Small SL-attention,
shared-policy configuration, exercised with PPO and DQN. Do not launch a separate
Base/attention/concat performance matrix; observation simplification is future
research work. Necessary framework correctness, recovery and quality checks remain.

Old SmartSOM training results are not an equivalence target. Official algorithm,
architecture and random-trajectory differences are acceptable; do not add seed or
algorithm adaptations to force matching results. The performance baseline remains
direct RLlib/Tune on the same version, research configuration and resources.

Ray 2.60 episode replay constructs default_rng(seed=None), independently of the
global NumPy seed. Existing DQN evidence has identical first-iteration stored
observations/actions/rewards, then diverges after learning. This supports official
replay randomness as a cause, without asserting all other randomness is excluded.

For DQN training comparisons, require the same configured seeds, environment-step
count, full per-module learner-update inventory, iterations and checkpoint work;
both runs must reach the frozen physical-tick budget. Record physical ticks,
actual agent-step counts and overshoot separately instead of requiring identical
random trajectories. Their variation is real work variation: timing differences
must not be presented as pure framework overhead or acceleration.

Keep three serial alternating pairs and the five-percent median steady/end-to-end
limits. Record startup separately without removing it from end-to-end timing.
Necessary checkpoint, evaluation, control, search and resource-release checks are
not replaced by this narrower performance scope. This authorizes no formal
experiments, publication, merge or changes to running experiments.
