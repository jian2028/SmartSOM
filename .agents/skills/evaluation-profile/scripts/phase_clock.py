"""Temporary in-process measurement hooks; no simulator source or decisions changed."""

import time
from collections import defaultdict
from functools import wraps


class PhaseClock:
    def __init__(self):
        self.rows = defaultdict(
            lambda: {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0}
        )
        self.stack = []

    def begin(self, label):
        frame = [label, time.perf_counter(), 0.0]
        self.stack.append(frame)
        return frame

    def end(self, frame):
        elapsed = time.perf_counter() - frame[1]
        assert self.stack.pop() is frame
        row = self.rows[frame[0]]
        row["calls"] += 1
        row["inclusive_seconds"] += elapsed
        row["exclusive_seconds"] += elapsed - frame[2]
        if self.stack:
            self.stack[-1][2] += elapsed

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            frame = self.begin(label)
            try:
                return original(*args, **kwargs)
            finally:
                self.end(frame)

        setattr(owner, name, measured)

    def install(self, policy):
        import smartsom.observations.social as social
        from smartsom.algorithms.rule_actions import RuleMetrics
        from smartsom.engine.base_protocol import BaseProtocol
        from smartsom.learning.models import EncodedModel
        from smartsom.telemetry.research_metrics import ResearchMetrics
        from smartsom.traffic.rule_mover import SharedMover

        original = BaseProtocol.decisions
        clock = self

        @wraps(original)
        def measured_decisions(protocol):
            gen = original(protocol)
            first, action = True, None
            while True:
                frame = clock.begin("decision_generation")
                try:
                    request = next(gen) if first else gen.send(action)
                    first = False
                except StopIteration:
                    return
                finally:
                    clock.end(frame)
                action = yield request

        BaseProtocol.decisions = measured_decisions
        self.wrap(type(policy), "choose", "policy_choose")
        from smartsom.representations.base import batching

        if policy.learner.pack is batching.pack and hasattr(batching, "_pack_one"):
            # Preserve pack identity: replacing it activates the custom-packer
            # fallback and disables the native singleton evaluation shortcut.
            self.wrap(batching, "_pack_one", "policy_tensor_packing")
        else:
            # Older baselines without the single-request builder.
            self.wrap(policy.learner, "pack", "policy_tensor_packing")
        self.wrap(EncodedModel, "actor", "policy_model_forward")
        self.wrap(BaseProtocol, "make_request", "request_features")
        self.wrap(social, "project_social", "social_features")
        self.wrap(SharedMover, "moves", "traffic_mover")
        self.wrap(BaseProtocol, "advance", "physical_advance")
        self.wrap(BaseProtocol, "traffic_view", "traffic_view")
        for name in ("before", "after", "decision", "finish"):
            self.wrap(ResearchMetrics, name, "research_metrics")
        self.wrap(RuleMetrics, "add", "rule_metrics")

    def distribution(self, wall_seconds):
        """Disjoint top-level groups for the native episode, with nested hooks removed."""

        def inclusive(label):
            return self.rows[label]["inclusive_seconds"]

        groups = {
            "candidate_and_observation": inclusive("decision_generation"),
            "policy_including_packing": inclusive("policy_choose"),
            "traffic_including_view": inclusive("traffic_mover")
            + inclusive("traffic_view"),
            "physical_advance": self.rows["physical_advance"]["exclusive_seconds"],
            "metrics": inclusive("research_metrics") + inclusive("rule_metrics"),
        }
        groups["other"] = wall_seconds - sum(groups.values())
        return {
            key: {"seconds": value, "percent": 100 * value / wall_seconds}
            for key, value in groups.items()
        }
