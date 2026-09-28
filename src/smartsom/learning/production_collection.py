"""Algorithm-independent physical-time trajectory and replay bookkeeping."""

import copy
import math
import random


def physical_gae(rows, bootstrap, gamma, lam):
    """Rows retain owner order; physical dt controls both discount factors."""
    advantage, following = 0.0, bootstrap
    for row in reversed(rows):
        discount = gamma ** row["dt"]
        continuation = not row["terminated"]
        delta = row["reward"] + discount * following * continuation - row["value"]
        advantage = delta + discount * lam ** row["dt"] * continuation * advantage
        row["advantage"], row["return"] = advantage, advantage + row["value"]
        following = row["value"]
    return rows


def normalize_advantages(rows):
    active = [r["advantage"] for r in rows if r["actor_mask"]]
    if len(active) > 1:
        mean = sum(active) / len(active)
        std = math.sqrt(sum((v - mean) ** 2 for v in active) / len(active))
        for row in rows:
            if row["actor_mask"]:
                row["advantage"] = (row["advantage"] - mean) / max(std, 1e-8)
    return rows


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.rows, self.position = capacity, [], 0
        self.rng = random.Random(seed)

    def add(self, row):
        if len(self.rows) < self.capacity:
            self.rows.append(copy.deepcopy(row))
        else:
            self.rows[self.position] = copy.deepcopy(row)
        self.position = (self.position + 1) % self.capacity

    def sample(self, count):
        return self.rng.sample(self.rows, count)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "rows": self.rows,
            "position": self.position,
            "random": self.rng.getstate(),
        }

    def load_state_dict(self, value):
        if self.capacity != value["capacity"]:
            raise ValueError("replay capacity changed during resume")
        self.rows, self.position = copy.deepcopy(value["rows"]), value["position"]
        self.rng.setstate(value["random"])


class PhysicalCollector:
    def __init__(self, groups, gamma, *, central=False):
        self.groups, self.gamma, self.central = set(groups), gamma, central
        self.active, self.trajectories, self.pending = {}, {}, {}
        self.counts = {
            g: {
                "decisions": 0,
                "actor_packets": 0,
                "value_samples": 0,
                "training_samples": 0,
                "replay_samples": 0,
                "censored_truncations": 0,
            }
            for g in groups
        }

    def collect(
        self,
        env_id,
        episode,
        coordinator,
        outcome,
        policies,
        rewards,
        *,
        algorithm,
        terminated=False,
        truncated=False,
        before=None,
        bootstrap_inputs=None,
    ):
        records = [r for r in coordinator.records if r["group"] in self.groups]
        packets = {}
        for record in records:
            group = record["group"]
            owner = "central" if self.central else record["owner"]
            key = (env_id, episode, owner, group)
            packets.setdefault(key, []).append(record)
            self.counts[group]["decisions"] += 1
            self.active[key] = record["role"]
        view = coordinator.sim.protocol.public_view()
        for key, role in list(self.active.items()):
            if key[:2] != (env_id, episode):
                continue
            group, policy = key[-1], policies[key[-1]]
            conditional = packets.get(key, [])
            owner = (
                conditional[0]["owner"]
                if conditional
                else (next(iter(coordinator.sim.agvs)) if self.central else key[2])
            )
            next_value_input, bootstrap = policy.value_input(role, owner, view)
            value_input, initial_value = policy.value_input(role, owner, before or view)
            reward = rewards[group]
            if algorithm == "ppo":
                inputs = [r["model_input"] for r in conditional]
                actions = [
                    next(
                        i
                        for i, c in enumerate(r["candidates"])
                        if c["identity"] == r["candidate"]
                    )
                    for r in conditional
                ]
                row = {
                    "inputs": inputs,
                    "actions": actions,
                    "value_input": value_input,
                    "log_probability": sum(r["log_probability"] for r in conditional),
                    "value": initial_value,
                    "actor_mask": any(r["actor_mask"] for r in conditional),
                    "value_mask": True,
                    "reward": reward,
                    "dt": 1,
                    "terminated": terminated,
                    "bootstrap": 0.0 if terminated else bootstrap,
                }
                self.trajectories.setdefault(key, []).append(row)
                self.counts[group]["value_samples"] += 1
                self.counts[group]["training_samples"] += 1
                self.counts[group]["actor_packets"] += int(row["actor_mask"])
            else:
                for record in conditional:
                    current = record["model_input"]
                    if key in self.pending:
                        previous = self.pending.pop(key)
                        previous["next_input"], previous["terminated"] = current, False
                        self.trajectories.setdefault(key, []).append(previous)
                    self.pending[key] = {
                        "input": current,
                        "action": next(
                            i
                            for i, c in enumerate(record["candidates"])
                            if c["identity"] == record["candidate"]
                        ),
                        "reward": 0.0,
                        "dt": 0,
                    }
                if key in self.pending:
                    pending = self.pending[key]
                    pending["reward"] += self.gamma ** pending["dt"] * reward
                    pending["dt"] += 1
                    if terminated:
                        pending = self.pending.pop(key)
                        pending.update(next_input=next_value_input, terminated=True)
                        self.trajectories.setdefault(key, []).append(pending)
                    elif truncated:
                        # No next actual decision: do not invent a masked WAIT or
                        # silently turn a time limit into a terminal TD target.
                        pending = self.pending.pop(key)
                        next_input = (bootstrap_inputs or {}).get((key[2], group))
                        if next_input is not None:
                            pending.update(next_input=next_input, terminated=False)
                            self.trajectories.setdefault(key, []).append(pending)
                        else:
                            self.counts[group]["censored_truncations"] += 1
            if terminated or truncated:
                self.active.pop(key)
        return view

    def drain(self, algorithm, lam=0.95):
        groups = {g: [] for g in sorted(self.groups)}
        for key, rows in self.trajectories.items():
            if algorithm == "ppo":
                physical_gae(rows, rows[-1]["bootstrap"], self.gamma, lam)
            groups[key[-1]].extend(rows)
            if algorithm == "dqn":
                self.counts[key[-1]]["replay_samples"] += len(rows)
                self.counts[key[-1]]["training_samples"] += len(rows)
        self.trajectories = {}
        if algorithm == "ppo":
            for rows in groups.values():
                normalize_advantages(rows)
        return groups

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, value):
        vars(self).update(copy.deepcopy(value))
