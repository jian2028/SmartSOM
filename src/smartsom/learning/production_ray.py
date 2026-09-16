"""RLlib central and role-shared PPO over the same production decision adapter."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.agent_to_module_mapping import AgentToModuleMapping
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.postprocessing import Postprocessing

from smartsom.config.codec import primitive
from smartsom.config.training import framework_seed
from smartsom.learning.production_env import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ProductionEnv,
)
from smartsom.trace.production import atomic_json, seal_checkpoint, state_hash


class SemanticBatchOrder(ConnectorV2):
    """Keep every column in the same episode/semantic-agent order before batching."""

    def __call__(self, *, batch, episodes, **kwargs):
        ranks = {episode.id_: i for i, episode in enumerate(episodes)}
        for column, items in batch.items():
            if (
                isinstance(items, dict)
                and items
                and all(isinstance(key, tuple) and len(key) == 3 for key in items)
            ):
                batch[column] = {
                    key: items[key]
                    for key in sorted(items, key=lambda k: (ranks[k[0]], k[1], k[2]))
                }
        return batch


class ProductionPPOConfig(PPOConfig):
    def build_env_to_module_connector(self, *args, **kwargs):
        pipeline = super().build_env_to_module_connector(*args, **kwargs)
        pipeline.insert_before(AgentToModuleMapping, SemanticBatchOrder())
        return pipeline

    def build_learner_connector(self, *args, **kwargs):
        pipeline = super().build_learner_connector(*args, **kwargs)
        pipeline.insert_before(AgentToModuleMapping, SemanticBatchOrder())
        return pipeline


class ProductionModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        from smartsom.config.extensions import ExtensionSpec
        from smartsom.learning.training_extensions import (
            effective_network_extensions,
            uses_extension_network,
        )

        extensions = self.model_config.get("extensions")
        spec = (
            ExtensionSpec.model_validate_json(json.dumps(extensions))
            if extensions
            else None
        )
        if uses_extension_network(spec, self.observation_space["observations"]):
            self.model_config["extensions"] = effective_network_extensions(
                spec,
                self.model_config["provider"],
                self.model_config.get("hidden_sizes", [64, 64]),
            )
            from smartsom.learning.extension_tensors import (
                network_configuration,
                public_space,
            )
            from smartsom.learning.torch_extensions import build_actor_critic

            network = build_actor_critic(
                public_space(self.observation_space["observations"]),
                int(self.action_space.n),
                network_configuration(
                    self.model_config["extensions"],
                    self.model_config["provider"],
                    self.model_config.get("hidden_sizes", [64, 64]),
                ),
                self.model_config["provider"],
                role=self.model_config.get("role"),
            )
            self.actor, self.critic = network.actor, network.critic
            return

        def network(output):
            sizes = [
                self.observation_space["observations"].shape[0],
                *self.model_config.get("hidden_sizes", [64, 64]),
                output,
            ]
            layers = []
            for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
                layers.append(torch.nn.Linear(a, b))
                if i < len(sizes) - 2:
                    layers.append(torch.nn.Tanh())
            return torch.nn.Sequential(*layers)

        self.actor = network(self.action_space.n)
        self.critic = network(1)

    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS]
        logits = self.actor(observation["observations"])
        mask = observation["action_mask"]
        return {Columns.ACTION_DIST_INPUTS: logits.masked_fill(mask == 0, -1e9)}

    def compute_values(self, batch, embeddings=None):
        return self.critic(batch[Columns.OBS]["observations"]).squeeze(-1)

    def _forward_train(self, batch, **kwargs):
        old_logits = batch.get(Columns.ACTION_DIST_INPUTS)
        if old_logits is not None:
            valid = batch.get(
                Columns.LOSS_MASK,
                torch.ones(
                    old_logits.shape[0], dtype=torch.bool, device=old_logits.device
                ),
            ).bool()
            mask = batch[Columns.OBS]["action_mask"].bool()
            if not torch.equal((old_logits > -1e8)[valid], mask[valid]):
                raise ValueError(
                    "sampled policy support differs from learner mask: misaligned batch"
                )
        return self._forward(batch, **kwargs)


class PhysicalGAE(ConnectorV2):
    def __init__(self, gamma, lam, reward_scale=1.0):
        super().__init__()
        self.gamma, self.lam = gamma, lam
        self.reward_scale = reward_scale

    def __call__(self, *, rl_module, batch, **kwargs):
        empty_modules = []
        for key, data in batch.items():
            with torch.no_grad():
                values = rl_module[key].compute_values(data)
            times = data[Columns.OBS]["physical_tick"].flatten()
            dt = torch.cat((times[1:] - times[:-1], times.new_zeros(1))).clamp(min=0)
            terminal = data[Columns.TERMINATEDS].float()
            truncated = data[Columns.TRUNCATEDS].float()
            nxt = torch.cat((values[1:], values.new_zeros(1)))
            discounts = self.gamma**dt
            residuals = (
                data[Columns.REWARDS] * self.reward_scale
                + discounts * (1 - terminal) * nxt
                - values
            )
            advantages = torch.zeros_like(values)
            carry = values.new_zeros(())
            for i in reversed(range(len(values))):
                carry = (
                    residuals[i]
                    + discounts[i]
                    * (self.lam ** dt[i])
                    * (1 - terminal[i])
                    * (1 - truncated[i])
                    * carry
                )
                advantages[i] = carry
            valid = data.get(
                Columns.LOSS_MASK, torch.ones_like(values, dtype=torch.bool)
            ).bool()
            selected = advantages[valid]
            mean = selected.mean() if selected.numel() else advantages.new_zeros(())
            std = (
                selected.std(unbiased=False).clamp(min=1e-4)
                if selected.numel()
                else advantages.new_ones(())
            )
            data[Postprocessing.VALUE_TARGETS] = (advantages + values).detach()
            data[Postprocessing.ADVANTAGES] = ((advantages - mean) / std).detach()

            # Bootstrap rows are only needed for value targets. Leaving them in
            # sparse role batches can produce a minibatch with zero loss rows,
            # whose masked mean is 0/0 in RLlib's PPO learner.
            def without_bootstrap(value):
                if isinstance(value, dict):
                    return {k: without_bootstrap(v) for k, v in value.items()}
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim
                    and value.shape[0] == len(valid)
                ):
                    return value[valid]
                return value

            if valid.any():
                batch[key] = without_bootstrap(data)
            else:
                empty_modules.append(key)
        for key in empty_modules:
            del batch[key]
        return batch


class ProductionLearner(PPOTorchLearner):
    def before_gradient_based_update(self, *, timesteps):
        super().before_gradient_based_update(timesteps=timesteps)
        self.updated_modules = set()

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        result = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        if not torch.isfinite(result):
            raise ValueError(f"non-finite PPO loss for {module_id}")
        self.updated_modules.add(module_id)
        return result

    def _update_module_kl_coeff(self, *, module_id, config, kl_loss):
        # RLlib retains metric keys for roles absent from the current batch;
        # a cleared metric has value NaN. It is not a measured KL divergence.
        if module_id not in self.updated_modules:
            return
        if not np.isfinite(kl_loss):
            raise ValueError(f"non-finite measured KL for {module_id}")
        return super()._update_module_kl_coeff(
            module_id=module_id, config=config, kl_loss=kl_loss
        )

    def build(self):
        super().build()
        self._learner_connector.remove(GeneralAdvantageEstimation)
        self._learner_connector.append(
            PhysicalGAE(
                self.config.gamma,
                self.config.lambda_,
                getattr(self.config, "learner_reward_scale", 1.0),
            )
        )


def wrapped_space(adapter, role=None):
    return gym.spaces.Dict(
        {
            "observations": adapter.hooks.spaces[role]
            if adapter.hooks
            else adapter.observation_space,
            "action_mask": gym.spaces.Box(0, 1, (adapter.action_count,), np.float32),
            "physical_tick": gym.spaces.Box(0, np.inf, (1,), np.float32),
        }
    )


def wrap_observation(adapter, observation):
    return {
        "observations": observation,
        "action_mask": adapter.action_masks().astype(np.float32),
        "physical_tick": np.array([adapter.sim.tick], np.float32),
    }


class CentralProductionEnv(gym.Env):
    def __init__(self, config):
        self.adapter = ProductionEnv(
            config["scenario"],
            config["max_jobs"],
            episode_source=config.get("episode_source"),
            extensions=config.get("extensions"),
            provider="rllib.ppo",
            limits=config.get("limits"),
            time_scale=config.get("time_scale"),
            count_scale=config.get("count_scale"),
            initial_observations=config.get("initial_observations"),
        )
        self.observation_space = wrapped_space(self.adapter)
        self.action_space = self.adapter.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.adapter.reset(seed=seed, options=options)
        return wrap_observation(self.adapter, obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.adapter.step(action)
        return wrap_observation(self.adapter, obs), reward, done, truncated, info


def role_mapping(agent, *args, **kwargs):
    return agent.split(":", 1)[0] + "_policy"


class ResourceProductionEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.adapter = ProductionEnv(
            config["scenario"],
            config["max_jobs"],
            episode_source=config.get("episode_source"),
            extensions=config.get("extensions"),
            provider="rllib.resource_ppo",
            limits=config.get("limits"),
            time_scale=config.get("time_scale"),
            count_scale=config.get("count_scale"),
            initial_observations=config.get("initial_observations"),
        )
        f = config["scenario"].factory
        self.possible_agents = [
            *(f"agv:{a.agv_id}" for a in f.agvs),
            *(f"machine:{m.machine_id}" for m in f.machines),
            *(f"quality:{s.inspection_station_id}" for s in f.inspection_stations),
            *(f"buffer:{b.buffer_id}" for b in f.buffers if b.role != "system_output"),
            *(f"buffer:{s.inspection_station_id}" for s in f.inspection_stations),
        ]
        self.agents = list(self.possible_agents)
        if self.adapter.hooks:
            self.adapter.hooks.fallback_role = role_mapping(self.possible_agents[0])
        self.observation_spaces = {
            a: wrapped_space(self.adapter, role_mapping(a)) for a in self.agents
        }
        self.action_spaces = {a: self.adapter.action_space for a in self.agents}
        self.gamma = config["gamma"]
        self.starts, self.pending_rewards, self.seen = {}, {}, set()

    def _actor(self):
        return self.adapter.actor if self.adapter.current else self.possible_agents[0]

    def reset(self, *, seed=None, options=None):
        obs, info = self.adapter.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        self.starts, self.pending_rewards, self.seen = {}, {}, set()
        actor = self._actor()
        self.seen.add(actor)
        return {actor: wrap_observation(self.adapter, obs)}, {actor: info}

    def step(self, actions):
        actor = self._actor()
        if set(actions) != {actor}:
            raise ValueError("one active decision owner must supply an action")
        self.adapter.validate_action(actions[actor])
        before = self.adapter.sim.tick
        self.starts[actor] = before
        obs, reward, done, truncated, info = self.adapter.step(actions[actor])
        if self.adapter.sim.tick > before:
            for agent, start in self.starts.items():
                self.pending_rewards[agent] = self.pending_rewards.get(agent, 0) + (
                    self.gamma ** (before - start)
                ) * info["role_rewards"].get(role_mapping(agent), reward)
        if done or truncated:
            self.agents = []
            final = wrap_observation(self.adapter, obs)
            observations = {agent: final for agent in self.seen}
            if self.adapter.hooks:
                from smartsom.learning.production_extensions import numpy_observation

                observations = {
                    agent: {
                        **final,
                        "observations": numpy_observation(
                            self.adapter.hooks.runtime.spaces[
                                role_mapping(agent)
                            ].zeros()
                        ),
                    }
                    for agent in self.seen
                }
            rewards = {
                agent: self.pending_rewards.get(agent, 0.0) for agent in self.seen
            }
            return (
                observations,
                rewards,
                {**dict.fromkeys(self.seen, done), "__all__": done},
                {**dict.fromkeys(self.seen, truncated), "__all__": truncated},
                {a: info for a in self.seen},
            )
        nxt = self._actor()
        self.seen.add(nxt)
        rewards = {nxt: self.pending_rewards.pop(nxt, 0.0)}
        return (
            {nxt: wrap_observation(self.adapter, obs)},
            rewards,
            {"__all__": False},
            {"__all__": False},
            {nxt: info},
        )


def train_ray(*args, **kwargs):
    from smartsom.learning.training_state import backend_cleanup

    with backend_cleanup() as cleanup:
        return _train_ray(*args, **kwargs, cleanup=cleanup)


def _train_ray(
    scenario,
    algorithm,
    directory,
    *,
    total_steps=16384,
    rollout_steps=256,
    resume_from=None,
    initialize_from=None,
    on_update=None,
    runtime=None,
    episode_source=None,
    sampling_spec=None,
    evidence=None,
    probe_only=False,
    cleanup,
):
    import json
    import random

    import cloudpickle

    if initialize_from is not None:
        from smartsom.experiments.packaging import _checkpoint_files
        from smartsom.learning.production_contract import validate_checkpoint_manifest

        validate_checkpoint_manifest(
            _checkpoint_files(Path(initialize_from)), scenario, algorithm
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(runtime.numerical_threads if runtime else 1)
    resource = algorithm.provider == "rllib.resource_ppo"
    env_config = {
        "scenario": scenario,
        "max_jobs": algorithm.max_jobs,
        "gamma": algorithm.gamma,
        "episode_source": episode_source,
        "extensions": algorithm.extensions,
        "time_scale": algorithm.time_scale,
        "count_scale": algorithm.count_scale,
        "initial_observations": json.loads(
            (Path(initialize_from) / "extension_state.json").read_text()
        )
        if initialize_from and algorithm.extensions
        else None,
    }
    probe = (
        ResourceProductionEnv(env_config)
        if resource
        else CentralProductionEnv(env_config)
    )
    if resource:
        representatives = {role_mapping(a): a for a in probe.possible_agents}
        specs = {
            role: RLModuleSpec(
                module_class=ProductionModule,
                observation_space=probe.observation_spaces[agent],
                action_space=probe.action_spaces[agent],
                model_config={
                    "hidden_sizes": list(algorithm.hidden_sizes),
                    "extensions": primitive(algorithm.extensions),
                    "provider": algorithm.provider,
                    "role": role,
                },
            )
            for role, agent in representatives.items()
        }
        spec = MultiRLModuleSpec(rl_module_specs=specs)
    else:
        specs = {
            "default_policy": RLModuleSpec(
                module_class=ProductionModule,
                observation_space=probe.observation_space,
                action_space=probe.action_space,
                model_config={
                    "hidden_sizes": list(algorithm.hidden_sizes),
                    "extensions": primitive(algorithm.extensions),
                    "provider": algorithm.provider,
                    "role": None,
                },
            )
        }
        spec = specs["default_policy"]
    config = (
        ProductionPPOConfig()
        .environment(
            ResourceProductionEnv if resource else CentralProductionEnv,
            env_config=env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=runtime.num_envs if runtime else 1,
            rollout_fragment_length=rollout_steps
            // (runtime.num_envs if runtime else 1),
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=1 if runtime and runtime.device == "cuda" else 0,
            learner_class=ProductionLearner,
        )
        .training(
            gamma=algorithm.gamma,
            lr=algorithm.learning_rate,
            train_batch_size_per_learner=rollout_steps,
            minibatch_size=min(algorithm.batch_size, rollout_steps),
            num_epochs=algorithm.n_epochs,
            entropy_coeff=algorithm.entropy_coefficient,
            lambda_=algorithm.gae_lambda,
            clip_param=algorithm.clip_range,
        )
        .rl_module(rl_module_spec=spec)
        .debugging(seed=framework_seed(scenario.seed, algorithm.provider))
    )
    if resource:
        config = config.multi_agent(
            policies={
                key: (None, s.observation_space, s.action_space, {})
                for key, s in specs.items()
            },
            policy_mapping_fn=role_mapping,
        )
    config.learner_reward_scale = algorithm.learner_reward_scale

    class LocalPPO(PPO):
        def _setup_logdir(self):
            self._logdir = str(directory / "rllib")
            Path(self._logdir).mkdir(exist_ok=True)

    import ray

    if ray.is_initialized():
        raise RuntimeError("production training requires its own Ray runtime")
    ray.init(address="local", num_cpus=1, include_dashboard=False)
    cleanup.callback(ray.shutdown)
    try:
        trainer = LocalPPO(config=config)
    except BaseException:
        ray.shutdown()
        raise
    cleanup.callback(trainer.stop)
    pool = sampler = pool_state = None
    if sampling_spec is not None:
        from smartsom.learning.ray_sampling import QuotaSampler, RayPoolTrainingState
        from smartsom.learning.sampling import OrderedSamplingPool

        evidence.bind()
        pool = OrderedSamplingPool(
            sampling_spec, runtime.num_envs, runtime.sampling_processes, evidence
        )
        cleanup.callback(pool.close)
        sampler = QuotaSampler(trainer, pool, evidence, resource)
        pool_state = RayPoolTrainingState(trainer, sampler, specs)
    if probe_only:
        from smartsom.learning.backend_probe import rllib_probe

        return rllib_probe(trainer, runtime)
    sampler_fields = (
        "env",
        "_ongoing_episodes",
        "_done_episodes_for_metrics",
        "_ongoing_episodes_for_metrics",
        "_cached_to_module",
        "_shared_data",
        "_needs_initial_reset",
        "_seed",
    )
    previous_steps = 0
    if resume_from is not None:
        source = Path(resume_from)
        if pool_state:
            from smartsom.learning.training_state import load_state, restore_rng

            pool_state.restore(source / "training")
            restore_rng(load_state(source / "rng.pkl"))
        else:
            trainer.restore_from_path(str(source / "trainer"))
            with (source / "sampler.pkl").open("rb") as stream:
                saved = cloudpickle.load(stream)
            runner = trainer.env_runner
            runner.env.close()
            for key, value in saved["runner"].items():
                setattr(runner, key, value)
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
        previous_steps = json.loads((source / "checkpoint.json").read_text())["steps"]
        if evidence:
            evidence.sampled_steps = previous_steps
    manifest = {
        "schema": "smartsom.production-checkpoint/v1",
        "observation_contract": OBSERVATION_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "provider": algorithm.provider,
        "scenario": primitive(scenario),
        "algorithm": primitive(algorithm),
        "factory_hash": state_hash(primitive(scenario.factory)),
        "quality_probability_visibility": scenario.quality_probability_visibility,
        "status": "training",
        "requested_steps": total_steps,
        "rollout_steps": rollout_steps,
        "resumed_from": str(Path(resume_from).resolve()) if resume_from else None,
    }
    from smartsom.experiments.evidence import source_identity

    manifest["source"] = source_identity()
    metadata_path = directory / ("training.json" if on_update else "checkpoint.json")
    atomic_json(metadata_path, manifest)
    if initialize_from is not None:
        weights = {}
        for key in specs:
            module = trainer.get_module(key)
            module.load_state_dict(
                torch.load(Path(initialize_from) / f"{key}.pt", weights_only=True)
            )
            weights[key] = module.get_state()
        trainer.learner_group.set_weights(weights)
    initial = {
        key: {
            k: v.detach().clone()
            for k, v in trainer.get_module(key).state_dict().items()
        }
        for key in specs
    }
    from smartsom.config.codec import digest
    from smartsom.learning.weights import weights_digest

    initial_hashes = {key: weights_digest(value) for key, value in initial.items()}
    manifest["initial_weights_sha256"] = (
        json.loads((Path(resume_from) / "checkpoint.json").read_text())[
            "initial_weights_sha256"
        ]
        if resume_from
        else digest(initial_hashes)
        if resource
        else next(iter(initial_hashes.values()))
    )

    def save(destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        changed, component_changes = {}, {}
        for key in specs:
            module = trainer.get_module(key)
            if any(not torch.isfinite(v).all() for v in module.state_dict().values()):
                raise RuntimeError(f"non-finite trained parameters in {key}")
            changed[key] = any(
                not torch.equal(initial[key][k], v)
                for k, v in module.state_dict().items()
            )
            component_changes[key] = {
                component: any(
                    not torch.equal(initial[key][name], value)
                    for name, value in module.state_dict().items()
                    if name.startswith(component + ".")
                )
                for component in ("actor", "critic")
            }
            torch.save(module.state_dict(), destination / f"{key}.pt")
        trainer.save_to_path(str(destination / "trainer"))
        if pool_state:
            from smartsom.learning.training_state import dump_state, rng_state

            pool_state.save(destination / "training")
            dump_state(destination / "rng.pkl", rng_state())
        vector = trainer.env_runner.env.unwrapped
        factories = getattr(vector, "env_fns", None)
        if factories is not None:
            vector.env_fns = ()
        try:
            with (destination / "sampler.pkl").open("wb") as stream:
                cloudpickle.dump(
                    {
                        "runner": {
                            key: getattr(trainer.env_runner, key)
                            for key in sampler_fields
                        },
                        "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                    },
                    stream,
                )
        finally:
            if factories is not None:
                vector.env_fns = factories
        final_hashes = {
            key: weights_digest(trainer.get_module(key).state_dict()) for key in specs
        }
        atomic_json(
            destination / "checkpoint.json",
            {
                **manifest,
                "status": "completed",
                "steps": previous_steps + completed,
                "environment_steps": previous_steps + completed,
                "final_weights_sha256": digest(final_hashes)
                if resource
                else next(iter(final_hashes.values())),
                "weights": final_hashes,
                "learner_updates": (previous_steps + completed) // rollout_steps,
                "parameters_changed": changed,
                "component_changes": component_changes,
                "modules": list(specs),
            },
        )

        if algorithm.extensions:
            extension_state = (
                pool.snapshots[0].extension_state["current"]
                if pool
                else vector.envs[0].unwrapped.adapter.hooks.runtime.state_dict()
            )
            atomic_json(destination / "extension_state.json", extension_state)
        seal_checkpoint(destination)

    try:
        completed = 0
        while completed < total_steps:
            if sampler:
                from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

                samples = sampler.sample(rollout_steps // runtime.num_envs)
                trainer.learner_group.foreach_learner(
                    lambda learner: learner._log_trainable_parameters()
                )
                results = trainer.learner_group.update(
                    episodes=samples,
                    timesteps={
                        "num_env_steps_sampled_lifetime": evidence.sampled_steps
                    },
                    num_epochs=algorithm.n_epochs,
                    minibatch_size=algorithm.batch_size,
                    shuffle_batch_per_epoch=trainer.config.shuffle_batch_per_epoch,
                )
                trainer.env_runner_group.sync_weights(
                    from_worker_or_learner_group=trainer.learner_group,
                    inference_only=True,
                )
                trainer._iteration = (previous_steps + completed) // rollout_steps + 1
                result = {"learners": MetricsLogger.peek_results(results[0])}
            else:
                result = trainer.train()
            completed += rollout_steps
            if on_update:
                metrics = {}

                def collect(data, prefix=""):
                    for key, value in data.items():
                        name = f"{prefix}/{key}" if prefix else key
                        if isinstance(value, dict):
                            collect(value, name)
                        elif isinstance(value, (float, int)) and np.isfinite(value):
                            metrics[name] = value

                collect(result.get("learners", {}))
                if on_update(
                    previous_steps + completed,
                    (previous_steps + completed) // rollout_steps,
                    metrics,
                    save,
                ):
                    break
        if on_update is None:
            save(directory)
        else:
            atomic_json(
                metadata_path,
                {
                    **manifest,
                    "status": "completed",
                    "steps": previous_steps + completed,
                },
            )
    except BaseException as exc:
        manifest.update(status="failed", failure=str(exc))
        atomic_json(metadata_path, manifest)
        raise
    return directory
