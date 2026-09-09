"""Optional SB3 Contrib backend over the shared, mask-aware Gym adapter."""

from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from smartsom.learning.gymnasium import extension_numpy
from smartsom.learning.training_extensions import (
    effective_network_extensions,
    inference_runtime_state,
    learner_scale,
    uses_extension_network,
)


def load_predictor(path: Path, *, deterministic=True, seed=None):
    from smartsom.learning.training_state import isolated_rng

    torch.set_num_threads(1)
    with isolated_rng():
        model = MaskablePPO.load(path / "model.zip", device="cpu")
    rng = np.random.default_rng(seed)

    def predict(view):
        observation = extension_numpy(view.observations)
        mask = np.asarray(view.action_mask, dtype=np.bool_)
        with torch.no_grad():
            distribution = model.policy.get_distribution(
                model.policy.obs_to_tensor(observation)[0],
                action_masks=mask.reshape(1, -1),
            )
            if not torch.isfinite(distribution.distribution.probs).all():
                raise ValueError("nonfinite checkpoint action probabilities")
            if not deterministic:
                probabilities = (
                    distribution.distribution.probs[0].cpu().numpy().astype(float)
                )
                probabilities[~mask] = 0
                return int(
                    rng.choice(
                        len(probabilities), p=probabilities / probabilities.sum()
                    )
                )
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        return int(action)

    predict.extension_state = inference_runtime_state(path)
    return predict


def train(resolved, env, evidence, checkpoint: Path, *, lifecycle=None):
    if lifecycle and (
        lifecycle.controls.num_envs > 1 or lifecycle.controls.sampling_processes
    ):
        from smartsom.learning.sampling import OrderedSamplingPool
        from smartsom.learning.sb3_sampling import OrderedVecEnv

        with OrderedSamplingPool(
            resolved,
            lifecycle.controls.num_envs,
            lifecycle.controls.sampling_processes,
            evidence,
        ) as pool:
            return _train(
                resolved, OrderedVecEnv(pool), evidence, checkpoint, lifecycle=lifecycle
            )
    if resolved.algorithm.algorithm.extensions and learner_scale(resolved) != 1.0:
        from smartsom.learning.sb3_reward_scaling import LearnerRewardEnv

        env = LearnerRewardEnv(env, learner_scale(resolved))
    return _train(resolved, env, evidence, checkpoint, lifecycle=lifecycle)


def _train(resolved, env, evidence, checkpoint: Path, *, lifecycle=None):
    from smartsom.learning.weights import weights_digest

    torch.set_num_threads(lifecycle.controls.numerical_threads if lifecycle else 1)
    spec = resolved.algorithm.algorithm.parameters

    class ManagedPPO(MaskablePPO):
        def train(self):
            super().train()
            metrics = evidence.learner(self._n_updates, self.logger.name_to_value)
            lifecycle.after_update(metrics)

        def collect_rollouts(self, *args, **kwargs):
            if lifecycle.stopped:
                return False
            return super().collect_rollouts(*args, **kwargs)

    policy, policy_kwargs = (
        "MlpPolicy",
        {"net_arch": list(spec.hidden_sizes), "activation_fn": torch.nn.Tanh},
    )
    extensions = resolved.algorithm.algorithm.extensions
    if uses_extension_network(extensions, env.observation_space):
        from smartsom.learning.sb3_extensions import ExtensionMaskableActorCriticPolicy

        policy = ExtensionMaskableActorCriticPolicy
        policy_kwargs = {
            "extensions": effective_network_extensions(
                extensions, "sb3.maskable_ppo", spec.hidden_sizes
            ),
            "provider": "sb3.maskable_ppo",
            "role": None,
            "fallback_hidden_sizes": list(spec.hidden_sizes),
        }
    model = (ManagedPPO if lifecycle else MaskablePPO)(
        policy,
        env,
        device=lifecycle.controls.device if lifecycle else "cpu",
        seed=resolved.framework_seed,
        learning_rate=spec.learning_rate,
        n_steps=spec.n_steps // (lifecycle.controls.num_envs if lifecycle else 1),
        batch_size=spec.batch_size,
        n_epochs=spec.n_epochs,
        gamma=spec.gamma,
        gae_lambda=spec.gae_lambda,
        clip_range=spec.clip_range,
        ent_coef=spec.entropy_coefficient,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    initial = weights_digest(model.policy.state_dict())
    if lifecycle:
        from smartsom.learning.training_state import SB3TrainingState

        if hasattr(env, "pool"):
            from smartsom.learning.sb3_sampling import SB3PoolTrainingState

            state = SB3PoolTrainingState(model, env)
        else:
            state = SB3TrainingState(model, env.unwrapped)
        initial = lifecycle.attach(state)["policy"]

    class Progress(BaseCallback):
        def _on_step(self):
            evidence.sampled_steps = self.num_timesteps
            evidence.progress("sampling")
            return True

        def _on_rollout_start(self):
            if model._n_updates and not lifecycle:
                evidence.learner(model._n_updates, model.logger.name_to_value)

    options = (
        {"reset_num_timesteps": False}
        if lifecycle and lifecycle.controls.resume_from
        else {}
    )
    model.learn(
        total_timesteps=resolved.run.budget.environment_steps - model.num_timesteps,
        callback=Progress(),
        **options,
    )
    if not lifecycle:
        evidence.learner(model._n_updates, model.logger.name_to_value)
    if model.num_timesteps != resolved.run.budget.environment_steps and not (
        lifecycle and lifecycle.stopped
    ):
        raise ValueError("SB3 exceeded the declared sampling budget")
    final = weights_digest(model.policy.state_dict())
    if lifecycle is None:
        model.save(checkpoint / "model.zip")
        restored = MaskablePPO.load(checkpoint / "model.zip", device="cpu")
        if weights_digest(restored.policy.state_dict()) != final:
            raise ValueError("SB3 checkpoint restoration changed parameters")
    return initial, final, model.num_timesteps, model._n_updates
