"""Algorithm-independent physical-time trajectory and replay bookkeeping."""

import copy
import math
import random

from smartsom.learning.compact_replay import SCHEMA, ReplayRows, StoredRow


def physical_gae(rows, bootstrap, gamma, lam):
    """Rows retain owner order; physical dt controls both discount factors."""
    advantage, following = 0.0, bootstrap
    for row in reversed(rows):
        discount = gamma ** row["dt"]
        continuation = not row["terminated"]
        delta = row["reward"] + discount * following * continuation - row["value"]
        advantage = delta + discount * lam ** row["dt"] * continuation * advantage
        row["advantage"], row["return"] = advantage, advantage + row["value"]
        row["raw_advantage"] = advantage
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
        if type(capacity) is not int or capacity < 1:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity, self._rows, self.position = capacity, [], 0
        self.rng = random.Random(seed)
        self.insertions = 0

    @property
    def rows(self):
        return ReplayRows(self)

    def add(self, row):
        insertion = self.insertions + 1
        stored = StoredRow.encode(dict(row, diagnostic_insertion=insertion))
        if len(self._rows) < self.capacity:
            self._rows.append(stored)
        else:
            self._rows[self.position] = stored
        self.insertions = insertion
        self.position = (self.position + 1) % self.capacity

    def sample(self, count):
        # Keep the same list population, slot order and random.sample algorithm.
        return [row.decode() for row in self.rng.sample(self._rows, count)]

    def state_dict(self):
        return {
            "schema": SCHEMA,
            "capacity": self.capacity,
            "insertions": self.insertions,
            "rows": [copy.deepcopy(row) for row in self._rows],
            "position": self.position,
            "random": self.rng.getstate(),
        }

    def load_state_dict(self, value):
        if not isinstance(value, dict):
            raise ValueError("invalid replay state")
        if type(value.get("capacity")) is not int or self.capacity != value["capacity"]:
            raise ValueError("replay capacity changed during resume")
        schema = value.get("schema")
        if schema not in (None, SCHEMA):
            raise ValueError("unsupported replay state schema")
        rows, position = value.get("rows"), value.get("position")
        insertions = value.get("insertions", 0)
        if (
            type(rows) is not list
            or len(rows) > self.capacity
            or type(position) is not int
            or not 0 <= position < self.capacity
            or (len(rows) < self.capacity and position != len(rows))
            or type(insertions) is not int
            or insertions < 0
        ):
            raise ValueError("invalid replay ring state")
        restored = []
        for row in rows:
            if schema is None:
                if type(row) is not dict:
                    raise ValueError("invalid legacy replay row")
                restored.append(StoredRow.encode(row))
            else:
                if type(row) is not StoredRow:
                    raise ValueError("invalid compact replay row")
                row.validate()
                restored.append(copy.deepcopy(row))
        rng = random.Random()
        try:
            rng.setstate(value["random"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid replay random state") from error
        # Validation and allocation finish before replacing the live ring or RNG.
        self._rows, self.position = restored, position
        self.insertions, self.rng = insertions, rng


class PhysicalCollector:
    def __init__(self, groups, gamma, *, central=False):
        self.groups, self.gamma, self.central = set(groups), gamma, central
        self.active, self.trajectories, self.pending = {}, {}, {}
        self.decision_coverage = {
            g: {"choice_decisions": 0, "forced_decisions": 0} for g in groups
        }
        self.decision_coverage_complete = True
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
            field = "choice_decisions" if record["actor_mask"] else "forced_decisions"
            self.decision_coverage[group][field] += 1
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
                        "choice_decision": bool(record["actor_mask"]),
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
        if "decision_coverage" not in value:
            self.decision_coverage = {
                g: {"choice_decisions": 0, "forced_decisions": 0} for g in self.groups
            }
            self.decision_coverage_complete = False
