"""Optional SB3 Contrib backend over the shared, mask-aware Gym adapter."""

from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback


def load_predictor(path: Path):
    torch.set_num_threads(1)
    model = MaskablePPO.load(path / "model.zip", device="cpu")

    def predict(view):
        observation = np.asarray(view.observations, dtype=np.float32)
        mask = np.asarray(view.action_mask, dtype=np.bool_)
        with torch.no_grad():
            distribution = model.policy.get_distribution(
                torch.as_tensor(observation).unsqueeze(0),
                action_masks=mask.reshape(1, -1),
            )
            if not torch.isfinite(distribution.distribution.probs).all():
                raise ValueError("nonfinite checkpoint action probabilities")
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        return int(action)

    return predict


def train(resolved, env, evidence, checkpoint: Path):
    from smartsom.learning.weights import weights_digest

    torch.set_num_threads(1)
    spec = resolved.algorithm.algorithm.parameters
    model = MaskablePPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=resolved.framework_seed,
        learning_rate=spec.learning_rate,
        n_steps=spec.n_steps,
        batch_size=spec.batch_size,
        n_epochs=spec.n_epochs,
        gamma=spec.gamma,
        gae_lambda=spec.gae_lambda,
        clip_range=spec.clip_range,
        ent_coef=spec.entropy_coefficient,
        policy_kwargs={
            "net_arch": list(spec.hidden_sizes),
            "activation_fn": torch.nn.Tanh,
        },
        verbose=0,
    )
    initial = weights_digest(model.policy.state_dict())

    class Progress(BaseCallback):
        def _on_step(self):
            evidence.sampled_steps = self.num_timesteps
            evidence.progress("sampling")
            return True

        def _on_rollout_start(self):
            if model._n_updates:
                evidence.learner(model._n_updates, model.logger.name_to_value)

    model.learn(
        total_timesteps=resolved.run.budget.environment_steps, callback=Progress()
    )
    evidence.learner(model._n_updates, model.logger.name_to_value)
    if model.num_timesteps != resolved.run.budget.environment_steps:
        raise ValueError("SB3 exceeded the declared sampling budget")
    final = weights_digest(model.policy.state_dict())
    model.save(checkpoint / "model.zip")
    restored = MaskablePPO.load(checkpoint / "model.zip", device="cpu")
    if weights_digest(restored.policy.state_dict()) != final:
        raise ValueError("SB3 checkpoint restoration changed parameters")
    return initial, final, model.num_timesteps, model._n_updates
