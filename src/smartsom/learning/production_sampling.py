"""Grid streams for the existing ordered local/process sampling coordinator."""

import json
from dataclasses import dataclass

from smartsom.config.codec import digest, primitive
from smartsom.config.production import ProductionRecipe
from smartsom.config.training import episode_root


@dataclass(frozen=True)
class ProductionSamplingSpec:
    recipe: ProductionRecipe
    seed: int
    initial_observations_json: str | None = None

    @property
    def algorithm(self):
        return self.recipe.algorithm


@dataclass(frozen=True)
class GridDecision:
    action_index: int
    indices: tuple
    actions: tuple
    proposals: tuple
    physical_dt: int
    conflict_count: int = 0
    reward_values: dict | None = None
    learner_rewards: dict | None = None


def committed_actions(row):
    """Physical commands that actually survived joint-action arbitration."""
    actions = []
    for group, role in (
        ("agvs", "agv"),
        ("machines", "machine"),
        ("quality", "quality"),
    ):
        for resource, command in row["actions"][group]:
            if f"{role}:{resource}" in row["rejections"]:
                continue
            if (
                command == "WAIT"
                or isinstance(command, dict)
                and command["job_id"] is None
            ):
                continue
            actions.append((role, resource, command))
    return tuple(actions)


class GridStream:
    def __init__(self, resolved, stream_id, num_envs):
        from smartsom.learning.production_env import ProductionEnv

        self.resolved, self.stream_id, self.num_envs = resolved, stream_id, num_envs
        self.local_episode = -1
        self.resource = resolved.algorithm.provider == "rllib.resource_ppo"
        from smartsom.learning.episode import EpisodeLimits

        budget = json.loads(resolved.recipe.training_json)
        limits = EpisodeLimits(
            **{
                key: budget[key]
                for key in ("max_ticks", "max_decisions")
                if key in budget
            }
        )
        config = {
            "scenario": resolved.recipe.scenario,
            "max_jobs": resolved.algorithm.max_jobs,
            "gamma": resolved.algorithm.gamma,
            "episode_source": self.episode,
            "extensions": resolved.algorithm.extensions,
            "provider": resolved.algorithm.provider,
            "limits": limits,
            "time_scale": resolved.algorithm.time_scale,
            "count_scale": resolved.algorithm.count_scale,
            "initial_observations": json.loads(resolved.initial_observations_json)
            if resolved.initial_observations_json
            else None,
        }
        if self.resource:
            from smartsom.learning.production_ray import ResourceProductionEnv

            self.env = ResourceProductionEnv(config)
        elif resolved.algorithm.provider == "rllib.ppo":
            from smartsom.learning.production_ray import CentralProductionEnv

            self.env = CentralProductionEnv(config)
        else:
            self.env = ProductionEnv(
                config["scenario"],
                config["max_jobs"],
                episode_source=self.episode,
                extensions=config["extensions"],
                provider=config["provider"],
                limits=limits,
                time_scale=config["time_scale"],
                count_scale=config["count_scale"],
                initial_observations=config["initial_observations"],
            )
        self.adapter = getattr(self.env, "adapter", self.env)
        self.indices = []
        self.steps = []
        self.trace = []
        self.initial_extensions = None

    def episode(self, local):
        return self.resolved.recipe.episode(
            episode_root(self.resolved.seed, local * self.num_envs + self.stream_id)
        )

    def reset(self):
        self.local_episode += 1
        self.indices, self.steps, self.trace = [], [], []
        if self.adapter.hooks:
            self.initial_extensions = self.adapter.hooks.runtime.state_dict()
        return self.env.reset()

    def step(self, value):
        actor, tick, decision = (
            self.adapter.actor,
            self.adapter.sim.tick,
            self.adapter.decisions,
        )
        try:
            result = self.env.step(value)
        except BaseException as exc:
            # A wrapper or hook can raise after the core already committed.
            # Retain that progress once; do not fabricate a learner reward.
            if self.adapter.decisions > decision:
                try:
                    info = self.adapter.last_info
                    if not info or info.get("decision_index") != decision:
                        info = None
                    self._record(value, actor, self.adapter.sim.tick - tick, info, None)
                except Exception as evidence_error:
                    exc.add_note(
                        f"partial sampling evidence could not be saved: {evidence_error}"
                    )
            raise
        info = next(iter(result[-1].values())) if self.resource else result[-1]
        rewards = result[1] if self.resource else {"team": result[1]}
        learner_rewards = {
            key: float(reward) * self.resolved.algorithm.learner_reward_scale
            for key, reward in rewards.items()
        }
        self._record(value, actor, info["physical_dt"], info, learner_rewards)
        return result

    def _record(self, value, actor, physical, info, learner_rewards):
        self.indices.append(
            {key: int(item) for key, item in value.items()}
            if self.resource
            else int(value)
        )
        row = self.adapter.last_result if physical else None
        if row:
            self.trace.append(row)
        self.steps.append(
            GridDecision(
                action_index=int(
                    next(iter(value.values())) if self.resource else value
                ),
                indices=tuple((key, int(item)) for key, item in value.items())
                if self.resource
                else ((actor, int(value)),),
                actions=committed_actions(row) if row else (),
                proposals=(),
                physical_dt=physical,
                conflict_count=sum(
                    reason == "conflict" for reason in row["rejections"].values()
                )
                if row
                else 0,
                reward_values=primitive(info["reward_values"]) if info else None,
                learner_rewards=learner_rewards,
            )
        )

    def snapshot(self):
        from smartsom.learning.sampling import EpisodeSnapshot

        sim = self.adapter.sim
        return EpisodeSnapshot(
            self.local_episode * self.num_envs + self.stream_id,
            self.stream_id,
            self.local_episode,
            self.adapter.scenario,
            tuple(self.steps),
            tuple(self.trace),
            (),
            self.adapter.episode_reward,
            self.adapter.reason if sim else None,
            sim.snapshot() if sim else None,
            self.adapter.finished if sim else False,
            tuple(self.env.possible_agents) if self.resource else (),
            {
                "initial": self.initial_extensions,
                "current": self.adapter.hooks.runtime.state_dict(),
            }
            if self.adapter.hooks
            else None,
        )

    def state(self):
        return {
            "stream_id": self.stream_id,
            "local_episode": self.local_episode,
            "indices": primitive(self.indices),
            "snapshot_sha256": digest(self.snapshot()),
            "initial_extensions": self.initial_extensions,
        }

    def restore(self, state):
        if state["stream_id"] != self.stream_id:
            raise ValueError("sampling stream identity differs on restore")
        self.local_episode = state["local_episode"] - 1
        self.adapter.episode_index = state["local_episode"] - 1
        if self.adapter.hooks:
            self.adapter.hooks.runtime.load_state_dict(state["initial_extensions"])
        self.reset()
        for action in state["indices"]:
            self.step(action)
        if self.state() != state:
            raise ValueError("grid sampling replay differs from saved activity")
        return self.snapshot()

    def command(self, command, value):
        if command in ("step", "reset"):
            result = self.step(value) if command == "step" else self.reset()
            return result, self.snapshot()
        if command == "state":
            return self.state()
        if command == "restore":
            return self.restore(value)
        if command == "mask":
            return self.adapter.action_masks()
        if command == "spaces":
            if self.resource:
                return (
                    self.env.observation_spaces,
                    self.env.action_spaces,
                    self.env.possible_agents,
                )
            return self.env.observation_space, self.env.action_space, ()
        raise ValueError(f"unknown sampling operation {command}")
