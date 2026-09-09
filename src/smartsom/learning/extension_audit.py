"""Replay research streams separately from the unchanged raw physical ledger."""

from smartsom.config.codec import digest, primitive
from smartsom.learning.training_extensions import (
    environment_arguments,
    extension_step_record,
    restore_initial_state,
    verify_restored_state,
)


class ExtensionTrainingAudit:
    def __init__(self, resolved):
        self.resolved = resolved
        self.previous = {}

    def episode(self, inp, row):
        spec = self.resolved.algorithm.algorithm
        if spec.extensions is None:
            if "extension_state" in row or "extension_steps" in row:
                raise ValueError(
                    "unextended training contains research extension evidence"
                )
            return
        resource = spec.provider == "rllib.resource_ppo"
        if resource:
            from smartsom.learning.pettingzoo import SmartSOMParallelEnv

            kind = SmartSOMParallelEnv
        else:
            from smartsom.learning.gymnasium import SchedulingEnv

            kind = SchedulingEnv
        env = kind(
            inp,
            spec.projection,
            limits=self.resolved.run.budget.limits(),
            **environment_arguments(self.resolved),
        )
        try:
            saved = row["extension_state"]
            stream = row.get("stream_id", 0)
            previous = self.previous.get(stream, env.extensions.state_dict())
            if digest(previous) != digest(saved["episode_initial_state"]):
                raise ValueError(
                    "research extension state does not continue its stream history"
                )
            restore_initial_state(env, saved)
            env.reset()
            if len(row["extension_steps"]) != len(row["steps"]):
                raise ValueError("research reward/observation ledger coverage mismatch")
            for raw, expected in zip(row["steps"], row["extension_steps"], strict=True):
                env.step(dict(raw["indices"]) if resource else raw["index"])
                if primitive(extension_step_record(env.steps[-1])) != expected:
                    raise ValueError("research observation/reward replay mismatch")
            verify_restored_state(env, saved)
            self.previous[stream] = saved["runtime"]
        finally:
            env.close()
