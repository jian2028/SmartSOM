"""Optional local RLlib PPO; only the protocol differs from the Parallel API."""

from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.common.agent_to_module_mapping import AgentToModuleMapping
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from smartsom.config.codec import digest
from smartsom.learning.checkpoint import RoleWeights
from smartsom.learning.extension_tensors import observation_tensor
from smartsom.learning.pettingzoo import SmartSOMParallelEnv
from smartsom.learning.rllib import MaskedPPOModule
from smartsom.learning.training_extensions import (
    effective_network_extensions,
    environment_arguments,
    inference_runtime_state,
    learner_scale,
    stack_observations,
    uses_extension_network,
)
from smartsom.learning.weights import weights_digest


def policy_mapping(agent_id, *args, **kwargs):
    return SmartSOMParallelEnv.policy_for_agent(agent_id)


class SemanticBatchOrder(ConnectorV2):
    """Ray's episode iterator yields sets; stabilize rows before role batching.

    Apply the same permutation to EVERY column. Episode order is the supplied
    sampling order, never a UUID sort; within each episode use semantic agent ID.
    This changes neither values nor masks nor probability distributions.
    """

    def __call__(self, *, batch, episodes, **kwargs):
        ranks = {episode.id_: i for i, episode in enumerate(episodes)}
        for column, items in batch.items():
            if (
                isinstance(items, dict)
                and items
                and all(isinstance(k, tuple) and len(k) == 3 for k in items)
            ):
                batch[column] = {
                    key: items[key]
                    for key in sorted(items, key=lambda k: (ranks[k[0]], k[1], k[2]))
                }
        return batch


class ScaleLearnerRewards(ConnectorV2):
    """Scale copied training tensors, never episodes or semantic reward evidence."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, *, batch, **kwargs):
        return {
            role: {**data, Columns.REWARDS: data[Columns.REWARDS] * self.scale}
            for role, data in batch.items()
        }


class ResourcePPOConfig(PPOConfig):
    def __init__(self, algo_class=None, *, learner_reward_scale=1.0):
        super().__init__(algo_class=algo_class)
        self.learner_reward_scale = learner_reward_scale

    def build_env_to_module_connector(self, *args, **kwargs):
        pipeline = super().build_env_to_module_connector(*args, **kwargs)
        pipeline.insert_before(AgentToModuleMapping, SemanticBatchOrder())
        return pipeline

    def build_learner_connector(self, *args, **kwargs):
        pipeline = super().build_learner_connector(*args, **kwargs)
        pipeline.insert_before(AgentToModuleMapping, SemanticBatchOrder())
        if self.learner_reward_scale != 1.0:
            # PPOLearner.build appends GAE after this completed tensor pipeline.
            # Scaling here supplies BOTH advantages and critic targets in the
            # same units, including bootstrap predictions, without touching envs.
            pipeline.append(ScaleLearnerRewards(self.learner_reward_scale))
        return pipeline


class RLlibResourceEnv(MultiAgentEnv):
    """Space discovery is pure: construction never calls scientific reset()."""

    def __init__(self, config):
        super().__init__()
        resolved = config["resolved"]
        from smartsom.learning.extensions import install_registrations

        install_registrations(config.get("registrations", ()))
        self.parallel = SmartSOMParallelEnv(
            resolved.episode(0).input,
            resolved.algorithm.algorithm.projection,
            limits=resolved.run.budget.limits(),
            episode_source=lambda index: resolved.episode(index).input,
            **environment_arguments(resolved),
        )
        self.possible_agents = self.parallel.possible_agents.copy()
        self._agent_ids = set(self.possible_agents)
        self.agents = []
        self.observation_spaces = self.parallel.observation_spaces
        self.action_spaces = self.parallel.action_spaces
        self.step_progress = None

    def reset(self, *, seed=None, options=None):
        result = self.parallel.reset(seed=seed, options=options)
        self.agents = self.parallel.agents.copy()
        return result

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.parallel.step(actions)
        terminated["__all__"] = bool(terminated) and all(terminated.values())
        truncated["__all__"] = bool(truncated) and all(truncated.values())
        # RLlib validates terminal reward/observation keys against this list.
        # Keep the resources named in this final transition; __all__ terminates
        # them together. PettingZoo independently empties its live agent list.
        self.agents = self.possible_agents.copy()
        if self.step_progress:
            self.step_progress(self.parallel.steps[-1])
        return obs, rewards, terminated, truncated, infos

    def close(self):
        self.parallel.close()


def load_predictor(path: Path, manifest, *, deterministic=True, seed=None):
    from smartsom.learning.training_state import isolated_rng

    torch.set_num_threads(1)
    modules = {}
    rng = np.random.default_rng(seed)
    for weights in manifest.role_weights:
        with isolated_rng():
            module = RLModule.from_checkpoint(path / weights.role)
        if weights_digest(module.get_state()) != weights.final_sha256:
            raise ValueError("resource checkpoint restored weights disagree")
        module.eval()
        modules[weights.role] = module

    def predict(decision):
        indices = {}
        for role, module in modules.items():
            views = [v for v in decision.views if v.role == role]
            if not views:
                continue
            batch = {
                Columns.OBS: {
                    "observations": observation_tensor(
                        stack_observations([v.observations for v in views])
                    ),
                    "action_mask": torch.as_tensor(
                        np.asarray([v.action_mask for v in views], dtype=np.int8)
                    ),
                }
            }
            with torch.no_grad():
                logits = module.forward_inference(batch)[Columns.ACTION_DIST_INPUTS]
            for row, (view, index) in enumerate(
                zip(views, logits.argmax(-1).tolist(), strict=True)
            ):
                if deterministic:
                    indices[view.agent_id] = int(index)
                else:
                    probabilities = (
                        torch.softmax(logits[row], dim=-1).cpu().numpy().astype(float)
                    )
                    probabilities[np.logical_not(view.action_mask)] = 0
                    indices[view.agent_id] = int(
                        rng.choice(
                            len(probabilities), p=probabilities / probabilities.sum()
                        )
                    )
        return indices

    predict.extension_state = inference_runtime_state(path)
    return predict


def train(resolved, env, evidence, checkpoint: Path, *, lifecycle=None):
    torch.set_num_threads(lifecycle.controls.numerical_threads if lifecycle else 1)
    p = resolved.algorithm.algorithm.parameters
    streams = lifecycle.controls.num_envs if lifecycle else 1
    parallel = lifecycle and (streams > 1 or lifecycle.controls.sampling_processes)
    representatives = {policy_mapping(a): a for a in env.possible_agents}
    extended = resolved.algorithm.algorithm.extensions
    custom_roles = {
        role
        for role, agent in representatives.items()
        if uses_extension_network(
            extended, env.observation_space(agent)["observations"]
        )
    }
    env_config = {"resolved": resolved}
    if extended:
        from smartsom.learning.extensions import export_registrations

        env_config["registrations"] = export_registrations(
            extended, "rllib.resource_ppo"
        )
    if custom_roles:
        from smartsom.learning.rllib_extensions import ExtensionPPOTorchRLModule
    specs = {
        role: RLModuleSpec(
            module_class=ExtensionPPOTorchRLModule
            if role in custom_roles
            else MaskedPPOModule,
            observation_space=env.observation_space(agent),
            action_space=env.action_space(agent),
            model_config={
                "extensions": effective_network_extensions(
                    extended, "rllib.resource_ppo", p.hidden_sizes
                ),
                "provider": "rllib.resource_ppo",
                "role": role,
                "fallback_hidden_sizes": list(p.hidden_sizes),
            }
            if role in custom_roles
            else {
                "fcnet_hiddens": list(p.hidden_sizes),
                "fcnet_activation": p.activation,
            },
        )
        for role, agent in sorted(representatives.items())
    }
    config = (
        ResourcePPOConfig(learner_reward_scale=learner_scale(resolved))
        .environment(
            env=RLlibResourceEnv,
            env_config=env_config,
            # Ray's check resets and samples unmasked actions. The dedicated
            # protocol tests use separate instances; live scientific episodes
            # must start at zero and only consume real learner proposals.
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=streams,
            rollout_fragment_length=p.n_steps // streams,
            batch_mode="truncate_episodes",
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=int(
                bool(lifecycle and lifecycle.controls.device == "cuda")
            ),
        )
        .multi_agent(
            policies={
                role: (None, spec.observation_space, spec.action_space, {})
                for role, spec in specs.items()
            },
            policy_mapping_fn=policy_mapping,
            count_steps_by="env_steps",
        )
        .training(
            gamma=p.gamma,
            lambda_=p.gae_lambda,
            lr=p.learning_rate,
            clip_param=p.clip_range,
            entropy_coeff=p.entropy_coefficient,
            train_batch_size_per_learner=p.n_steps,
            minibatch_size=p.batch_size,
            num_epochs=p.n_epochs,
        )
        .debugging(seed=resolved.framework_seed)
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=specs))
    )
    log_dir = str(evidence.run_dir / "rllib")

    class LocalPPO(PPO):
        def _setup_logdir(self):
            self._logdir = log_dir
            Path(log_dir).mkdir(exist_ok=True)

    if ray.is_initialized():
        raise RuntimeError("training requires its own local Ray runtime")
    algorithm = None
    try:
        ray.init(address="local", num_cpus=1, include_dashboard=False)
        algorithm = LocalPPO(config=config)
        if parallel:
            from smartsom.learning.ray_sampling import train_streams

            return train_streams(
                algorithm, resolved, evidence, lifecycle, specs, resource=True
            )
        wrapper = algorithm.env_runner.env.unwrapped.envs[0].unwrapped
        if not isinstance(wrapper, RLlibResourceEnv):
            raise ValueError("RLlib did not construct the Parallel protocol adapter")
        active = wrapper.parallel
        if active.episode_index != -1:
            raise ValueError("framework construction consumed a scientific episode")
        active.on_episode = evidence.episode
        evidence.active_env = active

        def sampled(step):
            evidence.sampled_steps += 1
            evidence.agent_steps += len(step.indices)
            evidence.physical_actions += len(step.actions)
            evidence.conflicts += sum(
                r.disposition in ("job_claimed", "no_longer_feasible")
                for r in step.proposals
            )
            evidence.progress("sampling")

        wrapper.step_progress = sampled
        initial = {
            role: weights_digest(algorithm.get_module(role).get_state())
            for role in specs
        }
        if lifecycle:
            from smartsom.learning.training_state import RayTrainingState

            initial = lifecycle.attach(RayTrainingState(algorithm, active, specs))
        initial_actor = {
            role: weights_digest(
                (
                    algorithm.get_module(role).network.actor
                    if role in custom_roles
                    else algorithm.get_module(role).pi
                ).state_dict()
            )
            for role in specs
        }
        updates, count = evidence.updates, evidence.sampled_steps
        while count < resolved.run.budget.environment_steps and not (
            lifecycle and lifecycle.stopped
        ):
            evidence.progress("learning", force=True)
            algorithm.learner_group.foreach_learner(
                lambda learner: learner._log_trainable_parameters()
            )
            metrics = algorithm.train()
            count = int(metrics["num_env_steps_sampled_lifetime"])
            updates += 1
            if (
                count != updates * p.n_steps
                or count != evidence.sampled_steps
                or evidence.agent_steps != count * len(active.possible_agents)
            ):
                raise ValueError("resource PPO joint/agent sampling budget mismatch")
            for role in specs:
                weights_digest(algorithm.get_module(role).get_state())
            numeric = evidence.learner(updates, metrics["learners"])
            if lifecycle:
                algorithm.env_runner.num_env_steps_sampled_lifetime = count
                lifecycle.after_update(numeric)
        final = {}
        for role in specs:
            module = algorithm.get_module(role)
            final[role] = weights_digest(module.get_state())
            if (
                final[role] == initial[role]
                or weights_digest(
                    (
                        module.network.actor if role in custom_roles else module.pi
                    ).state_dict()
                )
                == initial_actor[role]
            ):
                raise ValueError(f"{role} has no actual actor parameter update")
            if lifecycle is None:
                module.save_to_path(checkpoint / role)
                restored = RLModule.from_checkpoint(checkpoint / role)
                if weights_digest(restored.get_state()) != final[role]:
                    raise ValueError(
                        f"{role} checkpoint restoration changed parameters"
                    )
        evidence.role_weights = tuple(
            RoleWeights(role=r, initial_sha256=initial[r], final_sha256=final[r])
            for r in sorted(specs)
        )
        return digest(initial), digest(final), count, updates
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
