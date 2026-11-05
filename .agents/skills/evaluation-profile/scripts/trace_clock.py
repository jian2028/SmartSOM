"""Separate, expensive exact trajectory digest; never use this mode for timings."""

import hashlib
import json
from dataclasses import asdict
from functools import wraps


class TraceDigest:
    def __init__(self):
        self.hashes = {
            key: hashlib.sha256() for key in ("requests_actions", "transitions")
        }
        self.counts = dict.fromkeys(self.hashes, 0)

    def add(self, key, row):
        self.hashes[key].update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        self.counts[key] += 1

    def install(self, policy):
        from smartsom.engine.production import ProductionSimulator

        original_choose = type(policy).choose
        original_step = ProductionSimulator.step
        trace = self

        @wraps(original_choose)
        def choose(bound, request, *args, **kwargs):
            action = original_choose(bound, request, *args, **kwargs)
            trace.add("requests_actions", (asdict(request), action))
            return action

        @wraps(original_step)
        def step(core, *args, **kwargs):
            result = original_step(core, *args, **kwargs)
            trace.add(
                "transitions",
                {
                    "result": result,
                    "tick": core.tick,
                    "jobs": core.jobs,
                    "agvs": core.agvs,
                    "machines": core.machine_state,
                    "storage": core.storage,
                    "completed": sorted(core.completed),
                    "released": sorted(core.released),
                    "queue": core.queue,
                    "shipment_times": core.shipment_times,
                    "qualified": sorted(core._qualified_shipments),
                    "overdue": core._overdue_time,
                    "reward": core.total_reward,
                    "reward_components": core._last_reward_components,
                    "metrics": dict(core.metrics),
                    "protocol_metrics": dict(core.protocol.metrics),
                },
            )
            return result

        type(policy).choose = choose
        ProductionSimulator.step = step

    def result(self):
        return {
            key: {"count": self.counts[key], "sha256": value.hexdigest()}
            for key, value in self.hashes.items()
        }
