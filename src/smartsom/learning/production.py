"""Training and checkpoint inference for the canonical production adapter."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from smartsom.config.codec import primitive
from smartsom.config.training import framework_seed
from smartsom.learning.production_contract import (
    validate_checkpoint_manifest,
)
from smartsom.learning.production_contract import (
    validate_model_contract as validate_model_contract,
)
from smartsom.learning.production_env import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ProductionEnv,
)
from smartsom.trace.production import atomic_json, seal_checkpoint, state_hash


def physical_advantages(
    rewards, values, starts, deltas, last_values, dones, gamma, lam
):
    """GAE in physical time; zero-duration decision phases have unit discount."""
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(last_values)
    for step in reversed(range(len(rewards))):
        alive = 1.0 - (dones if step == len(rewards) - 1 else starts[step + 1])
        nxt = last_values if step == len(rewards) - 1 else values[step + 1]
        discount = gamma ** deltas[step]
        residual = rewards[step] + discount * nxt * alive - values[step]
        carry = residual + discount * (lam ** deltas[step]) * alive * carry
        advantages[step] = carry
    return advantages, advantages + values


def train_sb3(*args, **kwargs):
    from smartsom.learning.training_state import backend_cleanup

    with backend_cleanup() as cleanup:
        return _train_sb3(*args, **kwargs, cleanup=cleanup)


def _train_sb3(
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
    import random

    import cloudpickle
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import (
        MaskableDictRolloutBuffer,
        MaskableRolloutBuffer,
    )
    from stable_baselines3.common.callbacks import BaseCallback

    if initialize_from is not None:
        from smartsom.experiments.packaging import _checkpoint_files
        from smartsom.learning.production_contract import validate_checkpoint_manifest

        validate_checkpoint_manifest(
            _checkpoint_files(Path(initialize_from)), scenario, algorithm
        )

    class PhysicalBuffer(MaskableRolloutBuffer):
        def reset(self):
            super().reset()
            self.physical_deltas = np.ones((self.buffer_size, self.n_envs), np.float32)

        def compute_returns_and_advantage(self, last_values, dones):
            self.advantages, self.returns = physical_advantages(
                self.rewards,
                self.values,
                self.episode_starts,
                self.physical_deltas,
                last_values.clone().cpu().numpy().flatten(),
                dones,
                self.gamma,
                self.gae_lambda,
            )

    class RecordTime(BaseCallback):
        def _on_step(self):
            buffer = self.model.rollout_buffer
            buffer.physical_deltas[buffer.pos] = [
                i["physical_dt"] for i in self.locals["infos"]
            ]
            if pool is None:
                self.locals["rewards"] *= algorithm.learner_reward_scale
            if evidence:
                evidence.sampled_steps = self.num_timesteps
                evidence.progress("sampling")
            return True

    torch.set_num_threads(runtime.numerical_threads if runtime else 1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pool = None
    if sampling_spec is not None:
        from smartsom.learning.sampling import OrderedSamplingPool
        from smartsom.learning.sb3_sampling import OrderedVecEnv

        evidence.bind()
        pool = OrderedSamplingPool(
            sampling_spec, runtime.num_envs, runtime.sampling_processes, evidence
        )
        env = OrderedVecEnv(pool)
    else:
        env = ProductionEnv(
            scenario,
            algorithm.max_jobs,
            episode_source=episode_source,
            extensions=algorithm.extensions,
            provider=algorithm.provider,
            time_scale=algorithm.time_scale,
            count_scale=algorithm.count_scale,
            initial_observations=json.loads(
                (Path(initialize_from) / "extension_state.json").read_text()
            )
            if initialize_from and algorithm.extensions
            else None,
        )
    device = runtime.device if runtime else "cpu"
    cleanup.callback(env.close)
    if isinstance(env.observation_space, gym.spaces.Dict):

        class PhysicalDictBuffer(MaskableDictRolloutBuffer):
            def reset(self):
                super().reset()
                self.physical_deltas = np.ones(
                    (self.buffer_size, self.n_envs), np.float32
                )

            compute_returns_and_advantage = PhysicalBuffer.compute_returns_and_advantage

        buffer_class = PhysicalDictBuffer
    else:
        buffer_class = PhysicalBuffer

    class ManagedPPO(MaskablePPO):
        stopped = False

        def train(self):
            super().train()
            if on_update:
                self.stopped = bool(
                    on_update(
                        self.num_timesteps,
                        self._n_updates,
                        dict(self.logger.name_to_value),
                        save,
                    )
                )

        def collect_rollouts(self, *args, **kwargs):
            if self.stopped:
                return False
            return super().collect_rollouts(*args, **kwargs)

    from smartsom.learning.training_extensions import (
        effective_network_extensions,
        uses_extension_network,
    )

    policy, policy_kwargs = "MlpPolicy", {"net_arch": list(algorithm.hidden_sizes)}
    if uses_extension_network(algorithm.extensions, env.observation_space):
        from smartsom.learning.sb3_extensions import ExtensionMaskableActorCriticPolicy

        policy = ExtensionMaskableActorCriticPolicy
        policy_kwargs = {
            "extensions": effective_network_extensions(
                algorithm.extensions, algorithm.provider, algorithm.hidden_sizes
            )
        }
    model = ManagedPPO(
        policy,
        env,
        n_steps=rollout_steps // (runtime.num_envs if runtime else 1),
        batch_size=min(algorithm.batch_size, rollout_steps),
        n_epochs=algorithm.n_epochs,
        learning_rate=algorithm.learning_rate,
        gamma=algorithm.gamma,
        gae_lambda=algorithm.gae_lambda,
        clip_range=algorithm.clip_range,
        ent_coef=algorithm.entropy_coefficient,
        seed=framework_seed(scenario.seed, algorithm.provider),
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
        rollout_buffer_class=buffer_class,
    )
    if probe_only:
        from smartsom.learning.backend_probe import sb3_probe

        return sb3_probe(model, runtime)
    if resume_from is not None:
        source = Path(resume_from)
        with (source / "sampler.pkl").open("rb") as stream:
            saved = cloudpickle.load(stream)
        if pool is not None:
            pool.restore(saved["pool"])
            env.reset_infos, env._seeds, env._options = saved["vector"]
        model = ManagedPPO.load(
            source / "model.zip",
            env=env if pool is not None else saved["env"],
            device=device,
            force_reset=False,
        )
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
    elif initialize_from is not None:
        loaded = MaskablePPO.load(Path(initialize_from) / "model.zip", device=device)
        model.policy.load_state_dict(loaded.policy.state_dict(), strict=True)
    initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
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
        "resumed_from": str(Path(resume_from).resolve()) if resume_from else None,
        "rollout_steps": rollout_steps,
    }
    from smartsom.experiments.evidence import source_identity

    manifest["source"] = source_identity()
    from smartsom.learning.weights import weights_digest

    manifest["initial_weights_sha256"] = (
        json.loads((Path(resume_from) / "checkpoint.json").read_text())[
            "initial_weights_sha256"
        ]
        if resume_from
        else weights_digest(initial)
    )
    metadata_path = directory / ("training.json" if on_update else "checkpoint.json")
    atomic_json(metadata_path, manifest)

    def save(destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        model.save(destination / "model")
        with (destination / "sampler.pkl").open("wb") as stream:
            cloudpickle.dump(
                {
                    **(
                        {
                            "pool": pool.state(),
                            "vector": (env.reset_infos, env._seeds, env._options),
                        }
                        if pool is not None
                        else {"env": model.get_env()}
                    ),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                },
                stream,
            )
        atomic_json(
            destination / "checkpoint.json",
            {
                **manifest,
                "status": "completed",
                "steps": model.num_timesteps,
                "environment_steps": model.num_timesteps,
                "final_weights_sha256": weights_digest(model.policy.state_dict()),
                "weights": {"policy": weights_digest(model.policy.state_dict())},
                "learner_updates": model._n_updates,
            },
        )

        if algorithm.extensions:
            extension_state = (
                pool.snapshots[0].extension_state["current"]
                if pool
                else env.hooks.runtime.state_dict()
            )
            atomic_json(destination / "extension_state.json", extension_state)
        seal_checkpoint(destination)

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=RecordTime(),
            reset_num_timesteps=resume_from is None,
        )
        changed = any(
            not torch.equal(initial[k], v) for k, v in model.policy.state_dict().items()
        )
        if on_update is None:
            save(directory)
        manifest.update(
            status="completed",
            steps=model.num_timesteps,
            parameters_changed=changed,
            learner_updates=model._n_updates,
            environment_steps=model.num_timesteps,
            final_weights_sha256=weights_digest(model.policy.state_dict()),
            weights={"policy": weights_digest(model.policy.state_dict())},
        )
        atomic_json(metadata_path, manifest)
        if on_update is None:
            seal_checkpoint(directory)
    except BaseException as exc:
        manifest.update(status="failed", failure=str(exc))
        atomic_json(metadata_path, manifest)
        raise
    return directory


class LearnedProductionDriver:
    """The adapter owns the sole simulator; the learned model receives arrays only."""

    def __init__(
        self,
        directory,
        scenario,
        *,
        deterministic=True,
        seed=None,
        limits=None,
        observations=None,
    ):
        try:
            self._load(
                directory,
                scenario,
                deterministic=deterministic,
                seed=seed,
                limits=limits,
                observations=observations,
            )
        except BaseException as exc:
            if hasattr(self, "env"):
                try:
                    self.env.close()
                except BaseException as cleanup_error:
                    exc.add_note(
                        f"inference environment cleanup failed: {cleanup_error}"
                    )
            raise

    def _load(self, directory, scenario, *, deterministic, seed, limits, observations):
        directory = Path(directory)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        if directory.is_file():
            directory = directory.parent
        self.manifest = json.loads((directory / "checkpoint.json").read_text())
        validate_checkpoint_manifest(self.manifest, scenario)
        from smartsom.experiments.packaging import _checkpoint_files

        _checkpoint_files(directory)
        from smartsom.config.production import AlgorithmConfig

        algorithm = AlgorithmConfig.model_validate_json(
            json.dumps(self.manifest["algorithm"])
        )
        from smartsom.learning.episode import EpisodeLimits

        recipe_path = directory / "recipe.json"
        budget = (
            json.loads(recipe_path.read_text())["config"]["training"]
            if recipe_path.is_file()
            else {}
        )
        limits = EpisodeLimits(
            **{
                key: getattr(limits, key)
                if limits is not None
                else budget.get(key, getattr(EpisodeLimits(), key))
                for key in ("max_ticks", "max_decisions")
            }
        )
        self.env = ProductionEnv(
            scenario,
            algorithm.max_jobs,
            extensions=algorithm.extensions,
            provider=algorithm.provider,
            limits=limits,
            time_scale=algorithm.time_scale,
            count_scale=algorithm.count_scale,
            evidence=observations
            if observations is not None
            else (
                json.loads(recipe_path.read_text())["config"]
                .get("logging", {})
                .get("observations", "hash")
                if recipe_path.is_file()
                else "hash"
            ),
        )
        if self.env.hooks:
            state_path = directory / "extension_state.json"
            if not state_path.is_file():
                raise ValueError(
                    "extended checkpoint has no saved inference hook state"
                )
            self.env.hooks.runtime.load_state_dict(json.loads(state_path.read_text()))
        self.learning_contract = {
            "schema": "smartsom.grid-learning-evidence/v1",
            "algorithm": primitive(algorithm),
            "limits": primitive(limits),
            "observations": self.env.evidence,
            "initial_extensions": self.env.hooks.runtime.state_dict()
            if self.env.hooks
            else None,
        }
        self.observation, _ = self.env.reset()
        self.sim = self.env.sim
        if self.manifest["provider"] == "sb3.maskable_ppo":
            from sb3_contrib import MaskablePPO

            self.model = MaskablePPO.load(directory / "model.zip", device="cpu")
            self.modules = None
        else:
            import torch

            from smartsom.learning.production_ray import ProductionModule, wrapped_space

            self.model = None
            self.modules = {}
            for key in self.manifest["modules"]:
                module = ProductionModule(
                    observation_space=wrapped_space(
                        self.env, key if key != "default_policy" else None
                    ),
                    action_space=self.env.action_space,
                    model_config={
                        "hidden_sizes": algorithm.hidden_sizes,
                        "extensions": primitive(algorithm.extensions),
                        "provider": algorithm.provider,
                        "role": key if key != "default_policy" else None,
                    },
                )
                module.load_state_dict(
                    torch.load(
                        directory / f"{key}.pt", weights_only=True, map_location="cpu"
                    )
                )
                module.eval()
                self.modules[key] = module

    def next_tick(self):
        tick = self.sim.tick
        while self.sim.tick == tick and not self.env.finished:
            mask = self.env.action_masks()
            if mask.sum() == 1:
                action = np.flatnonzero(mask)[0]
            elif self.model is not None:
                import torch

                with torch.no_grad():
                    observation, _ = self.model.policy.obs_to_tensor(self.observation)
                    distribution = self.model.policy.get_distribution(
                        observation, action_masks=mask
                    )
                    logits = distribution.distribution.logits[0].cpu().numpy()
                    self.env.current_scores = [
                        float(score) if valid else None
                        for score, valid in zip(logits, mask, strict=True)
                    ]
                if self.deterministic:
                    action = int(logits.argmax())
                else:
                    probabilities = (
                        distribution.distribution.probs[0].cpu().numpy().astype(float)
                    )
                    action = self.rng.choice(
                        len(mask), p=probabilities / probabilities.sum()
                    )
            else:
                import torch
                from ray.rllib.core.columns import Columns

                from smartsom.learning.production_ray import wrap_observation

                key = (
                    "default_policy"
                    if "default_policy" in self.modules
                    else self.env.role + "_policy"
                )
                from smartsom.learning.training_extensions import (
                    torch_observation_batch,
                )

                observation = torch_observation_batch(
                    wrap_observation(self.env, self.observation)
                )
                with torch.no_grad():
                    logits = self.modules[key]._forward({Columns.OBS: observation})[
                        Columns.ACTION_DIST_INPUTS
                    ]
                    if self.deterministic:
                        action = int(logits.argmax(-1)[0])
                    else:
                        probabilities = (
                            logits.softmax(-1)[0].cpu().numpy().astype(float)
                        )
                        action = self.rng.choice(
                            len(mask), p=probabilities / probabilities.sum()
                        )
                    self.env.current_scores = [
                        float(score) if valid else None
                        for score, valid in zip(logits[0].cpu(), mask, strict=True)
                    ]
            self.observation, _, _, _, _ = self.env.step(int(action))
        return self.env.last_result if self.sim.tick > tick else None


def load_policy(directory, scenario):
    if directory is None:
        raise ValueError("evaluation requires a trained checkpoint")
    return LearnedProductionDriver(directory, scenario)
