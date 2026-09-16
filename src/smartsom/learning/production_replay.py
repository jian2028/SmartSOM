"""Replay actual learner inputs and hook state without loading policy weights."""

from smartsom.config.codec import digest, primitive
from smartsom.config.production import AlgorithmConfig
from smartsom.learning.episode import EpisodeLimits
from smartsom.learning.production_env import ProductionEnv


class GridLearningAudit:
    def __init__(self, scenario, contract):
        import json

        if contract.get("schema") != "smartsom.grid-learning-evidence/v1":
            raise ValueError("unknown learning evidence contract")
        algorithm = AlgorithmConfig.model_validate_json(
            json.dumps(contract["algorithm"])
        )
        self.env = ProductionEnv(
            scenario,
            algorithm.max_jobs,
            extensions=algorithm.extensions,
            provider=algorithm.provider,
            limits=EpisodeLimits(**contract["limits"]),
            time_scale=algorithm.time_scale,
            count_scale=algorithm.count_scale,
            evidence=contract["observations"],
        )
        initial = contract["initial_extensions"]
        if bool(self.env.hooks) != (initial is not None):
            raise ValueError("missing initial extension state")
        if self.env.hooks:
            self.env.hooks.runtime.load_state_dict(initial)
        self.env.reset()
        self.decisions = 0

    def decision(self, recorded):
        if self.env.finished:
            raise ValueError("learning decision after termination")
        self.env.step(recorded["selected_index"])
        if (
            self.env.last_result
            and self.env.last_result["tick"] == self.env.sim.tick
            and not self.env.decision_log
        ):
            actual = self.env.last_result["decisions"][-1]
        else:
            actual = self.env.decision_log[-1]
        keys = set(actual) - {"scores"}
        if set(recorded) - {"scores"} != keys or any(
            digest(recorded[key]) != digest(actual[key]) for key in keys
        ):
            raise ValueError(
                f"learning input/reward replay mismatch at decision {self.decisions}"
            )
        self.decisions += 1

    def append(self, row):
        decisions = row.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("missing learning decision evidence")
        before = self.env.sim.tick
        for decision in decisions:
            if self.env.sim.tick != before:
                raise ValueError("learning decisions exceed physical tick coverage")
            self.decision(decision)
        if self.env.sim.tick != before + 1 or digest(
            self.env.last_result["actions"]
        ) != digest(row["actions"]):
            raise ValueError("learning decisions do not cover committed actions")
        if digest(self.env.sim.snapshot()) != digest(row["state"]):
            raise ValueError("learning adapter physical state mismatch")

    def pending(self, decisions):
        before = self.env.sim.tick
        for decision in decisions:
            self.decision(decision)
            if self.env.sim.tick != before:
                raise ValueError("pending learning decisions commit an unrecorded tick")

    def result(self):
        return {
            "decisions": self.decisions,
            "reason": self.env.reason,
            "state_sha256": digest(self.env.hooks.runtime.state_dict())
            if self.env.hooks
            else None,
            "pending": primitive(self.env.pending_decisions),
        }
