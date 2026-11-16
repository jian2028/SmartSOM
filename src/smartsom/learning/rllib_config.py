"""Shared official AlgorithmConfig construction for direct and Tune execution.

Importing this module is cheap; optional frameworks are loaded only by the
builder. Research structure comes from the semantic configuration, while
``training`` contains official RLlib training() keyword arguments, never legacy
learner parameters.
"""

from copy import deepcopy
from functools import partial


def validate_training_parameters(**training):
    """Reject retired learner controls without importing optional frameworks."""
    retired = {
        "update_trigger",
        "update_every",
        "update_amount",
        "updates_per_transition",
        "warmup_transitions",
        "exploration",
        "epsilon_final",
        "epsilon_decay_transitions",
        "batch_size",
        "clip",
        "critic_initial_scale",
        "entropy",
        "epochs",
        "initialization",
        "learning_rate",
        "learning_reward_scale",
        "replay_capacity",
        "target_update_interval",
        "target_update_clock",
        "width",
        "return_estimator",
        "gae_lambda",
    }
    found = retired.intersection(training)
    if found:
        raise ValueError(
            "retired native learner parameters: "
            + ", ".join(sorted(found))
            + "; use Ray 2.60 PPOConfig/DQNConfig.training parameters"
        )


def validate_algorithm_training(algorithm, training):
    """Validate supported research heads and replay without importing RLlib."""
    validate_training_parameters(**training)
    if algorithm != "dqn":
        return
    if training.get("num_atoms", 1) != 1 or training.get("noisy", False):
        raise ValueError("research DQN heads require num_atoms=1 and noisy=False")
    if training.get("store_buffer_in_checkpoints", True) is not True:
        raise ValueError("DQN requires store_buffer_in_checkpoints=True")
    replay_type = (
        "smartsom.learning.rllib_dqn.RestorableMultiAgentPrioritizedEpisodeReplayBuffer"
    )
    if training.get("replay_buffer_config", {}).get("type", replay_type) != replay_type:
        raise ValueError("DQN requires the approved restorable official replay")


def validate_research_adapters(options):
    """Validate the supported scientific adapters without loading frameworks."""
    central = options.control == "centralized"
    namespace = "smartsom.controllers.e1a" if central else "smartsom.representations.v2"
    if central and (options.method != "ppo" or len(options.production.policies) != 1):
        raise ValueError("centralized E1a requires one official PPO policy")
    for policy in options.production.policies.values():
        if (
            policy.observation.target != namespace + ".observation.create_observation"
            or policy.network is None
            or policy.network.target != namespace + ".network.create_network"
        ):
            raise ValueError(
                "RLlib requires Base/SL V2 or centralized E1a research adapters"
            )
        if central:
            if (
                policy.critic_observation.target
                != namespace + ".observation.create_critic_observation"
                or policy.critic_observation.params != policy.observation.params
            ):
                raise ValueError(
                    "E1a actor and critic require the same public global observation scales"
                )
            if policy.split_stages or policy.frozen_stages or policy.role_params:
                raise ValueError(
                    "centralized E1a uses one trainable policy, without role stages"
                )
            params = policy.network.params
            if set(params) - {"advance_only_when_idle"} or not isinstance(
                params.get("advance_only_when_idle", False), bool
            ):
                raise ValueError(
                    "E1a network supports only boolean advance_only_when_idle"
                )


def ppo_config(*, scenario, options, seed, **training):
    return build_algorithm_config(
        scenario=scenario,
        options=options,
        seed=seed,
        algorithm="ppo",
        training=training,
    )


def dqn_config(*, scenario, options, seed, **training):
    return build_algorithm_config(
        scenario=scenario,
        options=options,
        seed=seed,
        algorithm="dqn",
        training=training,
    )


for _factory in (ppo_config, dqn_config):
    _factory.CONTRACT = "smartsom.rllib-algorithm/v1"
    _factory.PARAMETER_SHARING = ("shared", "independent")
    _factory.validate_parameters = validate_training_parameters


def _module_for_agent(agent_id, *args, mapping, **kwargs):
    return mapping[agent_id]


def build_algorithm_config(
    *,
    scenario,
    options,
    algorithm,
    seed,
    training=None,
    env_runners=None,
    learners=None,
    resources=None,
    reporting=None,
    training_worlds=None,
):
    """Build, but do not start, an official PPO/DQN Algorithm.

    Callers share this builder and may attach official evaluation/callback
    configuration before building. No Ray runtime is initialized here.
    """
    if algorithm not in {"ppo", "dqn"}:
        raise ValueError("algorithm must be ppo or dqn")
    validate_research_adapters(options)
    central = options.control == "centralized"
    if central and algorithm != "ppo":
        raise ValueError("centralized E1a requires official PPO")
    for policy in options.production.policies.values():
        if policy.kind != "learning":
            raise ValueError("non-learning RLlib modules are not connected yet")
        if policy.role_params:
            raise ValueError(
                "native role_params need explicit RLlib per-module migration"
            )
    training = deepcopy(training or {})
    validate_algorithm_training(algorithm, training)
    if algorithm == "dqn":
        replay = training.setdefault("replay_buffer_config", {})
        replay_type = (
            "smartsom.learning.rllib_dqn."
            "RestorableMultiAgentPrioritizedEpisodeReplayBuffer"
        )
        replay["type"] = replay_type
        training["store_buffer_in_checkpoints"] = True
    runner_settings = dict(env_runners or {})
    if runner_settings.get("num_envs_per_env_runner", 1) != 1:
        raise ValueError("MARL currently requires one factory per EnvRunner")
    runner_settings["num_envs_per_env_runner"] = 1

    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
        DefaultDQNTorchRLModule,
    )
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.policy.policy import PolicySpec

    from smartsom.learning.rllib_dqn import ReplayCheckpointDQN
    from smartsom.learning.rllib_env import FactoryMultiAgentEnv
    from smartsom.learning.rllib_metrics import FactoryProgress
    from smartsom.learning.rllib_models import ResearchDQNCatalog, ResearchPPOCatalog

    env_config = {
        "scenario": scenario,
        "options": options,
        "algorithm": algorithm,
        "action_seed": seed,
        "training_worlds": training_worlds,
    }
    if central:
        from smartsom.learning.rllib_central_env import (
            CentralFactoryEnv,
            CentralVectorEnv,
        )
        from smartsom.learning.rllib_central_models import (
            CentralPPOCatalog,
            central_env_connector,
            central_learner_connector,
        )

        environment = CentralVectorEnv
        probe = CentralFactoryEnv(env_config)
        try:
            module_spec = RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=probe.observation_space,
                action_space=probe.action_space,
                catalog_class=CentralPPOCatalog,
                model_config={
                    "owner_count": len(probe.owners),
                    "advance_only_when_idle": options.learning_policy.network.params.get(
                        "advance_only_when_idle", False
                    ),
                    "vf_share_layers": False,
                },
            )
        finally:
            probe.close()
        runner_settings.update(
            episodes_to_numpy=False,
            env_to_module_connector=central_env_connector,
            gym_env_vectorize_mode="vector_entry_point",
        )
    else:
        environment = FactoryMultiAgentEnv
        probe = FactoryMultiAgentEnv(env_config)
        specs, policies, trainable_modules = {}, {}, set()
        mapping = dict(probe.agent_modules)
        try:
            for agent, module_id in mapping.items():
                if module_id in specs:
                    continue
                group = probe.agent_groups[agent]
                codec = probe.codecs[group]
                role = group.split(".")[0]
                point = role + "." + codec.stages[0]
                policy_name = options.production.bindings.get(
                    point, options.production.default
                )
                if group not in options.production.policies[policy_name].frozen_stages:
                    trainable_modules.add(module_id)
                policies[module_id] = PolicySpec(
                    observation_space=probe.observation_spaces[agent],
                    action_space=probe.action_spaces[agent],
                )
                specs[module_id] = RLModuleSpec(
                    module_class=DefaultPPOTorchRLModule
                    if algorithm == "ppo"
                    else DefaultDQNTorchRLModule,
                    observation_space=probe.observation_spaces[agent],
                    action_space=probe.action_spaces[agent],
                    catalog_class=ResearchPPOCatalog
                    if algorithm == "ppo"
                    else ResearchDQNCatalog,
                    model_config={
                        "research_group": group,
                        "research_social": codec.settings.model_dump(exclude_none=True),
                        "research_stages": codec.stages,
                        "vf_share_layers": False,
                    },
                )
        finally:
            probe.close()
        module_spec = MultiRLModuleSpec(rl_module_specs=specs)
    config = (
        PPOConfig() if algorithm == "ppo" else DQNConfig(algo_class=ReplayCheckpointDQN)
    )
    if algorithm == "dqn":
        # RLlib replaces the replay dictionary when its type changes.
        replay_defaults = deepcopy(config.replay_buffer_config)
        replay_defaults.update(training.get("replay_buffer_config", {}))
        training["replay_buffer_config"] = replay_defaults
    config = (
        config.environment(
            environment,
            env_config=env_config,
            # RLlib's generic checker samples Discrete without consulting masks.
            disable_env_checking=True,
        )
        .env_runners(**runner_settings)
        .learners(**(learners or {}))
        .resources(**(resources or {}))
        .training(**training)
        .reporting(**(reporting or {}))
        .rl_module(rl_module_spec=module_spec)
        .fault_tolerance(
            restart_failed_env_runners=False,
            max_num_env_runner_restarts=0,
        )
        .debugging(seed=seed)
        .callbacks(callbacks_class=FactoryProgress)
    )
    if central:
        config.learners(learner_connector=central_learner_connector)
    else:
        config.multi_agent(
            policies=policies,
            policies_to_train=sorted(trainable_modules),
            policy_mapping_fn=partial(_module_for_agent, mapping=mapping),
        )
    return config
