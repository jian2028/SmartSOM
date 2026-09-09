"""Exact multi-stream quotas with RLlib's modules, connectors, Episodes and PPO learner.

Ray's default next-step vector collector can exceed its requested quota. This
collector resets completed streams explicitly, then executes exactly the same
number of physical transitions on each stream before the framework PPO update.
"""

import copy

from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.spaces.space_utils import unbatch

from smartsom.config.codec import digest
from smartsom.learning.checkpoint import RoleWeights
from smartsom.learning.sampling import OrderedSamplingPool
from smartsom.learning.training_state import (
    RayTrainingState,
    _ray_episode_state,
    dump_state,
    load_state,
)
from smartsom.learning.weights import weights_digest


class QuotaSampler:
    def __init__(self, algorithm, pool, evidence, resource):
        self.algorithm, self.pool, self.evidence, self.resource = (
            algorithm,
            pool,
            evidence,
            resource,
        )
        self.runner = algorithm.env_runner
        self.runner._shared_data = {}
        self.episodes = [None] * pool.num_envs
        self.started = False

    def reset(self, indices=None):
        results = self.pool.reset(indices)
        for index, (observation, info) in results.items():
            self.runner._new_episode(index, self.episodes)
            if self.resource:
                self.episodes[index].add_env_reset(observations=observation, infos=info)
            else:
                self.episodes[index].add_env_reset(observation=observation, infos=info)
        self.runner._ongoing_episodes = self.episodes
        self.runner._needs_initial_reset = False
        self.started = True

    def sample(self, steps_per_stream):
        if not self.started:
            self.reset()
        runner = self.runner
        completed = []
        for _ in range(steps_per_stream):
            # Fixed stream order also fixes role batching and policy RNG consumption.
            runner._shared_data["vector_env_episodes_map"] = {}
            batch = runner._env_to_module(
                episodes=self.episodes,
                batch={},
                explore=True,
                rl_module=runner.module,
                shared_data=runner._shared_data,
                metrics=runner.metrics,
            )
            outputs = runner.module.forward_exploration(
                batch, t=self.evidence.sampled_steps
            )
            outputs = runner._module_to_env(
                rl_module=runner.module,
                batch=outputs,
                episodes=self.episodes,
                explore=True,
                shared_data=runner._shared_data,
                metrics=runner.metrics,
            )
            actions = outputs.pop(Columns.ACTIONS)
            physical = outputs.pop(Columns.ACTIONS_FOR_ENV, actions)
            if not self.resource:
                actions, physical = unbatch(actions), unbatch(physical)
            results = self.pool.step(physical)
            reset = []
            for index, (observation, reward, terminated, truncated, info) in enumerate(
                results
            ):
                episode = self.episodes[index]
                if self.resource:
                    extra = {}
                    for column, values in outputs.items():
                        for agent, value in values[index].items():
                            extra.setdefault(agent, {})[column] = value
                    terminated = dict(
                        terminated,
                        __all__=bool(terminated) and all(terminated.values()),
                    )
                    truncated = dict(
                        truncated, __all__=bool(truncated) and all(truncated.values())
                    )
                    episode.add_env_step(
                        observations=observation,
                        actions=actions[index],
                        rewards=reward,
                        infos=info,
                        terminateds=terminated,
                        truncateds=truncated,
                        extra_model_outputs=extra,
                    )
                    step = self.pool.snapshots[index].steps[-1]
                    self.evidence.agent_steps += len(step.indices)
                    self.evidence.physical_actions += len(step.actions)
                    self.evidence.conflicts += sum(
                        proposal.disposition in ("job_claimed", "no_longer_feasible")
                        for proposal in step.proposals
                    )
                else:
                    episode.add_env_step(
                        observation=observation,
                        action=actions[index],
                        reward=reward,
                        infos=info,
                        terminated=terminated,
                        truncated=truncated,
                        extra_model_outputs={
                            column: values[index] for column, values in outputs.items()
                        },
                    )
                self.evidence.sampled_steps += 1
                if episode.is_done:
                    completed.append(episode.to_numpy())
                    reset.append(index)
            if reset:
                self.reset(reset)
            self.evidence.progress("sampling")
        continuations = [
            episode.cut(len_lookback_buffer=runner.config.episode_lookback_horizon)
            for episode in self.episodes
        ]
        pending = [episode.to_numpy() for episode in self.episodes if len(episode)]
        self.episodes = runner._ongoing_episodes = continuations
        samples = completed + pending
        if (
            sum(len(episode) for episode in samples)
            != steps_per_stream * self.pool.num_envs
        ):
            raise ValueError("RLlib stream sampler violated its exact global quota")
        runner.num_env_steps_sampled_lifetime = self.evidence.sampled_steps
        return samples


class RayPoolTrainingState(RayTrainingState):
    def __init__(self, algorithm, sampler, roles):
        super().__init__(algorithm, None, roles)
        self.sampler = sampler

    def save(self, directory):
        directory.mkdir()
        self.save_core(directory)
        dump_state(
            directory / "activity.pkl",
            {
                "streams": self.sampler.pool.state(),
                "episodes": [
                    _ray_episode_state(episode) for episode in self.sampler.episodes
                ],
                "shared": copy.deepcopy(self.sampler.runner._shared_data),
            },
        )

    def restore(self, directory):
        self.restore_core(directory)
        state = load_state(directory / "activity.pkl")
        self.sampler.pool.restore(state["streams"])
        kind = MultiAgentEpisode if self.sampler.resource else SingleAgentEpisode
        self.sampler.episodes = self.sampler.runner._ongoing_episodes = [
            kind.from_state(episode) for episode in state["episodes"]
        ]
        self.sampler.runner._shared_data = state["shared"]
        self.sampler.started = True


def train_streams(algorithm, resolved, evidence, lifecycle, roles, *, resource):
    controls, parameters = lifecycle.controls, resolved.algorithm.algorithm.parameters
    with OrderedSamplingPool(
        resolved, controls.num_envs, controls.sampling_processes, evidence
    ) as pool:
        sampler = QuotaSampler(algorithm, pool, evidence, resource)
        if getattr(evidence, "probe_only", False):
            from smartsom.learning.backend_probe import rllib_probe

            return rllib_probe(algorithm, controls)
        initial = lifecycle.attach(RayPoolTrainingState(algorithm, sampler, roles))
        while (
            evidence.sampled_steps < resolved.run.budget.environment_steps
            and not lifecycle.stopped
        ):
            samples = sampler.sample(parameters.n_steps // controls.num_envs)
            algorithm.learner_group.foreach_learner(
                lambda learner: learner._log_trainable_parameters()
            )
            results = algorithm.learner_group.update(
                episodes=samples,
                timesteps={"num_env_steps_sampled_lifetime": evidence.sampled_steps},
                num_epochs=parameters.n_epochs,
                minibatch_size=parameters.batch_size,
                shuffle_batch_per_epoch=algorithm.config.shuffle_batch_per_epoch,
            )
            algorithm.env_runner_group.sync_weights(
                from_worker_or_learner_group=algorithm.learner_group,
                inference_only=True,
            )
            updates = evidence.updates + 1
            algorithm._iteration = updates
            numeric = evidence.learner(updates, MetricsLogger.peek_results(results[0]))
            lifecycle.after_update(numeric)
        final = {
            role: weights_digest(algorithm.get_module(role).get_state())
            for role in roles
        }
        if any(initial[role] == final[role] for role in roles):
            raise ValueError("multi-stream training did not update every policy")
        if resource:
            evidence.role_weights = tuple(
                RoleWeights(
                    role=role, initial_sha256=initial[role], final_sha256=final[role]
                )
                for role in sorted(roles)
            )
        return (
            digest(initial) if resource else next(iter(initial.values())),
            digest(final) if resource else next(iter(final.values())),
            evidence.sampled_steps,
            evidence.updates,
        )
