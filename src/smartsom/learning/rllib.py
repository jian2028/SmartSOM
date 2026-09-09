"""Optional RLlib PPO module and driver using its public single-agent API."""

from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec

from smartsom.learning.gymnasium import SchedulingEnv


class RLlibSchedulingEnv(SchedulingEnv):
    """Serializable framework protocol; file writers are attached only locally."""

    def __init__(self, config):
        resolved = config["resolved"]
        super().__init__(
            resolved.episode(0).input,
            resolved.algorithm.algorithm.projection,
            limits=resolved.run.budget.limits(),
            observation_kind="masked",
            episode_source=lambda index: resolved.episode(index).input,
            strict_actions=True,
        )
        self.step_progress = None

    def step(self, action):
        result = super().step(action)
        if self.step_progress:
            self.step_progress()
        return result


class MaskedPPOModule(DefaultPPOTorchRLModule):
    """Mask after shared PPO encoding; masks never enter the value/actor encoder."""

    def __init__(self, *, observation_space, **kwargs):
        self.masked_space = observation_space
        super().__init__(observation_space=observation_space["observations"], **kwargs)
        self.observation_space = self.masked_space

    @staticmethod
    def _unpack(batch):
        if isinstance(batch[Columns.OBS], dict):
            obs = batch[Columns.OBS]
            return obs["action_mask"], {**batch, Columns.OBS: obs["observations"]}
        return batch["action_mask"], batch

    @staticmethod
    def _mask(output, mask):
        logits = output[Columns.ACTION_DIST_INPUTS]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite learner logits")
        output[Columns.ACTION_DIST_INPUTS] = logits.masked_fill(
            mask == 0, torch.finfo(logits.dtype).min
        )
        return output

    def _forward_inference(self, batch, **kwargs):
        mask, plain = self._unpack(batch)
        return self._mask(super()._forward_inference(plain, **kwargs), mask)

    def _forward_exploration(self, batch, **kwargs):
        mask, plain = self._unpack(batch)
        return self._mask(super()._forward_exploration(plain, **kwargs), mask)

    def _forward_train(self, batch, **kwargs):
        mask, plain = self._unpack(batch)
        return self._mask(super()._forward_train(plain, **kwargs), mask)

    def compute_values(self, batch, embeddings=None):
        _, plain = self._unpack(batch)
        return super().compute_values(plain, embeddings)


def load_predictor(path: Path, *, deterministic=True, seed=None):
    from smartsom.learning.training_state import isolated_rng

    torch.set_num_threads(1)
    with isolated_rng():
        module = RLModule.from_checkpoint(path / "module")
    module.eval()
    rng = np.random.default_rng(seed)

    def predict(view):
        batch = {
            Columns.OBS: {
                "observations": torch.as_tensor(
                    np.asarray(view.observations, dtype=np.float32)
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    view.action_mask, dtype=torch.float32
                ).unsqueeze(0),
            }
        }
        with torch.no_grad():
            result = module.forward_inference(batch)
        logits = result[Columns.ACTION_DIST_INPUTS]
        if deterministic:
            return int(logits.argmax(-1).item())
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(float)
        probabilities[np.logical_not(view.action_mask)] = 0
        return int(
            rng.choice(len(probabilities), p=probabilities / probabilities.sum())
        )

    return predict


def train(resolved, env, evidence, checkpoint: Path, *, lifecycle=None):
    from smartsom.learning.weights import weights_digest

    torch.set_num_threads(1)
    p = resolved.algorithm.algorithm.parameters

    config = (
        PPOConfig()
        .environment(
            env=RLlibSchedulingEnv,
            env_config={"resolved": resolved},
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=p.n_steps,
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
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
        .rl_module(
            rl_module_spec=RLModuleSpec(module_class=MaskedPPOModule),
            model_config={
                "fcnet_hiddens": list(p.hidden_sizes),
                "fcnet_activation": p.activation,
            },
        )
    )

    log_dir = str(evidence.run_dir / "rllib")

    class LocalPPO(PPO):
        def _setup_logdir(self):
            # Ray 2.58 removed logger_creator. Keep its Trainable scratch directory
            # inside this attempt without changing cwd or starting a Tune study.
            self._logdir = log_dir
            Path(self._logdir).mkdir(exist_ok=True)

    if ray.is_initialized():
        raise RuntimeError("training requires its own local Ray runtime")
    algorithm = None
    try:
        ray.init(address="local", num_cpus=1, include_dashboard=False)
        algorithm = LocalPPO(config=config)
        active = algorithm.env_runner.env.unwrapped.envs[0].unwrapped
        if not isinstance(active, RLlibSchedulingEnv):
            raise ValueError("RLlib did not construct the shared Gym adapter")
        active.on_episode = evidence.episode

        def sampled():
            evidence.sampled_steps += 1
            evidence.progress("sampling")

        active.step_progress = sampled
        evidence.active_env = active
        initial = weights_digest(algorithm.get_module().get_state())
        if lifecycle:
            from smartsom.learning.training_state import RayTrainingState

            initial = lifecycle.attach(
                RayTrainingState(algorithm, active, ["default_policy"])
            )["default_policy"]
        count = evidence.sampled_steps
        updates = evidence.updates
        while count < resolved.run.budget.environment_steps and not (
            lifecycle and lifecycle.stopped
        ):
            evidence.progress("learning", force=True)
            # Ray 2.58 logs these counts only when building the learner; its next
            # metrics reduction otherwise yields NaN for the cleared counters.
            # Refresh the actual counts instead of dropping or tolerating NaNs.
            algorithm.learner_group.foreach_learner(
                lambda learner: learner._log_trainable_parameters()
            )
            metrics = algorithm.train()
            count = int(metrics["num_env_steps_sampled_lifetime"])
            updates += 1
            if count != updates * p.n_steps:
                raise ValueError(f"RLlib sampling budget mismatch: {count}")
            evidence.sampled_steps = count
            weights_digest(algorithm.get_module().get_state())
            numeric = evidence.learner(updates, metrics["learners"])
            if lifecycle:
                lifecycle.after_update(numeric)
        module = algorithm.get_module()
        final = weights_digest(module.get_state())
        if lifecycle is None:
            module.save_to_path(checkpoint / "module")
            restored = RLModule.from_checkpoint(checkpoint / "module")
            if weights_digest(restored.get_state()) != final:
                raise ValueError("RLlib checkpoint restoration changed parameters")
        return initial, final, count, updates
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
