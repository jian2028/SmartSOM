"""Masked decision phases over one physical tick, for all learning backends.

Ranking draws are conditional categorical choices without replacement. Their
unchanged per-job features and shrinking masks implement a Plackett-Luce order.
Only the final physical commit advances time; callers must use physical_dt for
discounting rather than counting adapter requests as simulation ticks.
"""

from collections import deque

import gymnasium as gym
import numpy as np

from smartsom.config.codec import digest, primitive
from smartsom.domain.production import AGV_ACTIONS, JointCommand, MachineCommand
from smartsom.engine.production import ProductionSimulator
from smartsom.learning.production_contract import (
    ACTION_CONTRACT as ACTION_CONTRACT,
)
from smartsom.learning.production_contract import (
    OBSERVATION_CONTRACT as OBSERVATION_CONTRACT,
)
from smartsom.learning.production_contract import factory_identity


class ProductionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario,
        max_jobs=64,
        *,
        episode_source=None,
        extensions=None,
        provider="sb3.maskable_ppo",
        limits=None,
        time_scale=None,
        count_scale=None,
        initial_observations=None,
        evidence=None,
    ):
        self.scenario = scenario
        self.evidence = evidence
        self.episode_source = episode_source
        self.episode_index = -1
        self.limits = limits
        self.decisions = 0
        self.limit_hit = False
        self.max_jobs = max_jobs
        self.time_scale = time_scale or scenario.reward_time_scale
        self.count_scale = count_scale or max_jobs
        self.factory = scenario.factory
        self.factory_identity = factory_identity(scenario.factory)
        self.probability_visibility = scenario.quality_probability_visibility
        self.machine_ids = sorted(m.machine_id for m in self.factory.machines)
        self.agv_ids = sorted(a.agv_id for a in self.factory.agvs)
        self.types = sorted(self.factory.operation_types)
        self.owners = sorted(
            [b.buffer_id for b in self.factory.buffers]
            + [s.inspection_station_id for s in self.factory.inspection_stations]
            + self.machine_ids
            + self.agv_ids
        )
        self.max_modes = max(
            (len(m.quality_modes) for m in self.factory.machines), default=1
        )
        self.action_count = max(6, max_jobs * self.max_modes + 1)
        # Role/self, static map, current public resource summaries, candidate jobs.
        self.feature_count = 16 + self.factory.grid.width * self.factory.grid.height
        self.feature_count += len(self.agv_ids) * 4 + len(self.machine_ids) * 5
        self.feature_count += len(self.factory.ports) * (14 + len(self.types))
        self.feature_count += self.action_count * 14
        self.action_space = gym.spaces.Discrete(self.action_count)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.feature_count,), np.float32
        )
        self.sim = None
        self.requests = deque()
        self.current = None
        self.last_result = None
        self.last_info = None
        self.rankings = {}
        self.rank_remaining = {}
        self.pending = {}
        self.episode_reward = 0.0
        self.hooks = None
        if extensions is not None:
            from smartsom.learning.production_extensions import GridHooks

            self.hooks = GridHooks(self, extensions, provider)
            self.observation_space = next(iter(self.hooks.spaces.values()))
            if initial_observations is not None:
                self.hooks.runtime.initialize_observations(initial_observations)

    @property
    def actor(self):
        return self.current[0] if self.current else "terminal"

    @property
    def role(self):
        return self.actor.split(":", 1)[0]

    @property
    def finished(self):
        return self.limit_hit or self.sim.done

    @property
    def reason(self):
        return "budget_exhausted" if self.limit_hit else self.sim.status

    @property
    def pending_decisions(self):
        if self.last_result and self.last_result.get("decisions") is self.decision_log:
            return []
        return self.decision_log

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        from dataclasses import replace

        scenario = self.scenario if seed is None else replace(self.scenario, seed=seed)
        next_episode = self.episode_index + 1
        if self.episode_source is not None:
            scenario = self.episode_source(next_episode)
        if (
            factory_identity(scenario.factory) != self.factory_identity
            or scenario.quality_probability_visibility != self.probability_visibility
        ):
            raise ValueError(
                "episode source changed the frozen observation/action contract"
            )
        self.episode_index = next_episode
        self.scenario = scenario
        self.sim = ProductionSimulator(scenario)
        self.decisions, self.limit_hit = 0, False
        self.episode_reward = 0.0
        self.last_result = None
        self.last_info = None
        if self.hooks:
            self.hooks.runtime.begin_episode()
            self.hooks.cached = None
        self._begin()
        return self.observation(), {"tick": self.sim.tick, "actor": self.actor}

    def _begin(self):
        self.requests.clear()
        self.decision_log = []
        self.current_scores = None
        self.rankings, self.rank_remaining, self.pending = {}, {}, {}
        for owner in sorted(self.sim.rankings):
            jobs = self.sim.selectable(owner)
            if len(jobs) > self.max_jobs:
                raise ValueError(f"{owner} exceeds configured max_jobs={self.max_jobs}")
            self.rankings[owner] = []
            if len(jobs) > 1:
                self.rank_remaining[owner] = jobs
                self.requests.append((f"buffer:{owner}", tuple(jobs)))
            else:
                self.rankings[owner] = jobs
        self.phase = "ranking"
        self._next()

    def _physical_requests(self):
        self.view = self.sim.decision(self.rankings)
        self.phase = "physical"
        for key, choices in sorted(self.view["machine_choices"].items()):
            if len(choices) + 1 > self.action_count:
                raise ValueError("machine choices exceed configured encoding capacity")
            if choices:
                self.requests.append((f"machine:{key}", (None, *choices)))
        for key, ready in sorted(self.view["inspection_choices"].items()):
            if ready:
                self.requests.append((f"quality:{key}", ("WAIT", "START")))
        for key, mask in sorted(self.view["agv_masks"].items()):
            if sum(mask) > 1:
                self.requests.append((f"agv:{key}", AGV_ACTIONS))

    def _next(self):
        if not self.requests and self.phase == "ranking":
            self._physical_requests()
        self.current = self.requests.popleft() if self.requests else None

    def action_masks(self):
        mask = np.zeros(self.action_count, dtype=bool)
        if self.current is None:
            mask[0] = True
        elif self.role == "agv":
            mask[:6] = self.view["agv_masks"][self.actor.split(":", 1)[1]]
        elif self.role == "buffer":
            remaining = self.rank_remaining[self.actor.split(":", 1)[1]]
            mask[: len(self.current[1])] = [j in remaining for j in self.current[1]]
        else:
            mask[: len(self.current[1])] = True
        return mask

    def _command(self):
        agvs, machines, quality = [], [], []
        for actor, action in self.pending.items():
            role, key = actor.split(":", 1)
            if role == "agv":
                agvs.append((key, action))
            elif role == "machine":
                machines.append(
                    (key, MachineCommand(*action) if action else MachineCommand())
                )
            elif role == "quality":
                quality.append((key, action))
        return JointCommand(
            tuple(agvs),
            tuple(machines),
            tuple(quality),
            tuple((k, tuple(v)) for k, v in self.rankings.items()),
        )

    def validate_action(self, action):
        """Validate before wrappers mutate sampling or reward bookkeeping."""
        if self.finished:
            raise ValueError("step after termination")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < self.action_count
        ):
            raise ValueError("action index outside action space")
        if not self.action_masks()[action]:
            raise ValueError("masked action supplied to production adapter")

    def step(self, action):
        self.validate_action(action)
        actor, before = self.actor, self.sim.tick
        context = self.hooks.context() if self.hooks else None
        semantic = self.current[1][action] if self.current else "WAIT"
        entry = {
            "actor": actor,
            "candidates": list(self.current[1]) if self.current else ["WAIT"],
            "selected_index": int(action),
            "mask": self.action_masks().tolist(),
            "scores": self.current_scores,
        }
        if self.evidence:
            from smartsom.learning.extension_evidence import wire_observation

            observation = wire_observation(self.observation())
            entry.update(
                decision_index=self.decisions,
                tick=before,
                observations_sha256=digest(observation),
                state_before_sha256=digest(self.hooks.runtime.state_dict())
                if self.hooks
                else None,
            )
            if self.evidence == "full":
                entry["observations"] = primitive(observation)
        self.decision_log.append(entry)
        self.current_scores = None
        if self.current:
            selected = self.current[1][action]
            if self.role == "buffer":
                owner = actor.split(":", 1)[1]
                self.rankings[owner].append(selected)
                self.rank_remaining[owner].remove(selected)
                rest = self.rank_remaining[owner]
                if len(rest) > 1:
                    self.requests.appendleft((actor, self.current[1]))
                else:
                    self.rankings[owner].extend(rest)
            else:
                self.pending[actor] = selected
            self._next()
        reward = 0.0
        if self.current is None:
            self.last_result = self.sim.step(self._command())
            self.last_result["decisions"] = self.decision_log
            reward += self.last_result["reward"]
            if not self.sim.done:
                self._begin()
        self.episode_reward += reward
        self.decisions += 1
        if self.limits and not self.sim.done:
            self.limit_hit = (
                self.decisions >= self.limits.max_decisions
                or self.sim.tick >= self.limits.max_ticks
            )
        raw_reward = reward
        role_rewards = {}
        values, role_values = None, {}
        if self.hooks and (self.sim.tick > before or self.limit_hit):
            values, role_values = self.hooks.reward(context, semantic, reward)
            reward = values.learner
            role_rewards = {role: value.learner for role, value in role_values.items()}
        if self.hooks:
            self.hooks.cached = None
        terminated = self.sim.done and self.sim.status == "completed"
        truncated = self.sim.status == "truncated" or self.limit_hit
        info = {
            "physical_dt": self.sim.tick - before,
            "decision_index": self.decisions - 1,
            "tick": self.sim.tick,
            "actor": actor,
            "next_actor": self.actor,
            "passed": len(self.sim.completed),
            "status": self.sim.status,
            "end_reason": self.reason if self.finished else None,
            "raw_reward": raw_reward,
            "role_rewards": role_rewards,
            "reward_values": {
                "raw": raw_reward,
                "research": values.research if values else raw_reward,
                "learner": reward,
                "roles": primitive(role_values),
            },
        }
        self.last_info = info
        observation = self.observation()
        if self.evidence:
            entry.update(
                state_after_sha256=digest(self.hooks.runtime.state_dict())
                if self.hooks
                else None,
                reward={
                    "raw": raw_reward,
                    "research": values.research if values else raw_reward,
                    "learner": reward,
                    "roles": primitive(role_values),
                    "reason": self.reason,
                    "physical_dt": self.sim.tick - before,
                },
            )
        return observation, reward, terminated, truncated, info

    def observation(self):
        return self.hooks.encode() if self.hooks else self.raw_observation()

    def raw_observation(self):
        out = np.zeros(self.feature_count, np.float32)
        if self.current is None:
            return out
        view = self.sim.decision() if self.phase == "ranking" else self.view
        role, owner = self.actor.split(":", 1)
        out[["agv", "machine", "buffer", "quality"].index(role)] = 1
        out[4] = self.sim.tick / self.scenario.tick_limit
        out[5] = self.owners.index(owner) / max(1, len(self.owners))
        visible_jobs = set()
        origin = (0, 0)
        if role == "agv":
            vehicle = view["agvs"][owner]
            origin = tuple(vehicle["cell"])
            if vehicle["job"]:
                visible_jobs.add(vehicle["job"])
            for port in self.factory.ports:
                if max(abs(port.cell.x - origin[0]), abs(port.cell.y - origin[1])) <= 2:
                    for binding in port.bindings:
                        target, _ = self.sim._target(binding.target)
                        if target in view["storage"]:
                            visible_jobs.update(
                                j
                                for slot in view["storage"][target].values()
                                for j in slot
                            )
        else:
            stores = [owner]
            if role == "machine":
                stores += [self.sim.pre.get(owner), self.sim.post.get(owner)]
                job = view["machines"][owner]["job"]
                if job:
                    visible_jobs.add(job)
            for store in stores:
                if store in view["storage"]:
                    visible_jobs.update(
                        j for slot in view["storage"][store].values() for j in slot
                    )
        out[6:8] = [
            origin[0] / self.factory.grid.width,
            origin[1] / self.factory.grid.height,
        ]
        if role == "agv" and view["agvs"][owner]["job"]:
            cargo = view["jobs"][view["agvs"][owner]["job"]]
            out[8] = 1
            out[9] = (
                (self.types.index(cargo["next_operation"]) + 1) / (len(self.types) + 1)
                if cargo["next_operation"]
                else 1
            )
            out[10 + ["UNKNOWN", "PASS", "FAIL"].index(cargo["quality"])] = 1
            out[13:16] = [
                (cargo["due_at"] - self.sim.tick) / self.time_scale,
                cargo["priority"],
                cargo["risk"],
            ]
        elif role in ("buffer", "quality"):
            jobs = [view["jobs"][j] for j in visible_jobs]
            out[8:12] = [
                len(jobs) / self.count_scale,
                sum(j["quality"] == "UNKNOWN" for j in jobs) / self.count_scale,
                sum(j["quality"] == "PASS" for j in jobs) / self.count_scale,
                sum(j["quality"] == "FAIL" for j in jobs) / self.count_scale,
            ]
            out[12:16] = [
                max((self.sim.tick - j["since"] for j in jobs), default=0)
                / self.time_scale,
                min((j["due_at"] - self.sim.tick for j in jobs), default=0)
                / self.time_scale,
                sum(j["priority"] for j in jobs) / self.count_scale,
                sum(j["risk"] for j in jobs) / self.count_scale,
            ]
        cursor = 16
        for y in range(self.factory.grid.height):
            for x in range(self.factory.grid.width):
                out[cursor] = (x, y) not in self.sim.solids
                cursor += 1
        for key in self.agv_ids:
            row = view["agvs"][key]
            if (
                role == "agv"
                and max(
                    abs(row["cell"][0] - origin[0]), abs(row["cell"][1] - origin[1])
                )
                <= 2
            ):
                out[cursor : cursor + 4] = [
                    1,
                    row["cell"][0] / self.factory.grid.width,
                    row["cell"][1] / self.factory.grid.height,
                    bool(row["job"]),
                ]
            cursor += 4
        for key in self.machine_ids:
            row = view["machines"][key]
            if role == "machine" and owner == key:
                out[cursor : cursor + 5] = [
                    1,
                    row["status"] == "IDLE",
                    row["status"] == "PROCESSING",
                    row["down"],
                    row["elapsed"] / self.time_scale,
                ]
            cursor += 5
        port_roles = [
            "system_input",
            "machine_pre",
            "machine_post",
            "inspection",
            "system_output",
            "scrap",
            "machine",
        ]
        for port in sorted(self.factory.ports, key=lambda p: p.port_id):
            width = 14 + len(self.types)
            if not port.bindings:
                cursor += width
                continue
            target, _ = self.sim._target(port.bindings[0].target)
            kind = self.sim.roles.get(
                target, "scrap" if target in self.sim.scrap else "machine"
            )
            out[cursor : cursor + 2] = [
                (port.cell.x - origin[0]) / self.factory.grid.width,
                (port.cell.y - origin[1]) / self.factory.grid.height,
            ]
            if kind in port_roles:
                out[cursor + 2 + port_roles.index(kind)] = 1
            known = (
                role == "agv"
                and max(abs(port.cell.x - origin[0]), abs(port.cell.y - origin[1])) <= 2
            ) or target == owner
            out[cursor + 9] = known
            machine = (
                target
                if target in self.sim.machines
                else self.sim.buffers[target].machine_id
                if target in self.sim.buffers
                else None
            )
            if machine:
                for n, operation in enumerate(self.types):
                    out[cursor + 14 + n] = (
                        operation in self.sim.machines[machine].operation_types
                    )
            if known:
                if role == "agv":
                    out[cursor + 10] = (
                        self.view["interactions"][owner][port.port_id] is not None
                    )
                if target in view["storage"]:
                    used = sum(len(j) for j in view["storage"][target].values())
                    caps = list(self.sim.capacity[target].values())
                    infinite = any(c is None for c in caps)
                    out[cursor + 11] = infinite
                    out[cursor + 12] = (
                        1 if infinite else max(0, 1 - used / max(1, sum(caps)))
                    )
                    out[cursor + 13] = not self.sim._locked(target)
            cursor += width
        for i, action in enumerate(self.current[1]):
            base = cursor + i * 14
            out[base] = 1
            job = (
                action
                if role == "buffer"
                else action[0]
                if role == "machine" and action
                else None
            )
            if job and job in visible_jobs:
                row = view["jobs"][job]
                out[base + 1 : base + 8] = [
                    1,
                    row["step"] / max(1, len(self.types)),
                    (row["due_at"] - self.sim.tick) / self.time_scale,
                    row["priority"],
                    row["risk"],
                    self.sim.tick - row["since"],
                    row["location"] in self.sim.pre.values(),
                ]
                out[base + 8 + ["UNKNOWN", "PASS", "FAIL"].index(row["quality"])] = 1
                if role == "machine":
                    mode = next(
                        m
                        for m in self.sim.machines[owner].quality_modes
                        if m.quality_mode_id == action[1]
                    )
                    out[base + 11 : base + 13] = [
                        float(mode.time_scale),
                        float(mode.error_rate)
                        if self.scenario.quality_probability_visibility == "public"
                        else -1.0,
                    ]
                    out[base + 13] = (
                        row["machine_nominal_ticks"][owner] / self.time_scale
                    )
            elif role == "agv" and action == "INTERACT":
                transfer = self.sim.interaction(owner, self.view["rankings"])
                out[base + 1] = transfer is not None
        if not np.isfinite(out).all():
            raise ValueError("nonfinite policy observation")
        return out
