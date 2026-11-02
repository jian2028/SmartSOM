"""Per-episode utilisation and delivery timeline for evaluation reports.

Builds on the shared research metrics collector. An AGV counts as active while it
carries a job, and for the empty driving or service ticks that end in a pickup on
the same target. Empty ticks spent waiting, parking, or on a target that is later
abandoned count as idle.
"""

from collections import Counter, defaultdict

from smartsom.telemetry.research_metrics import ResearchMetrics


class EvaluationMetrics(ResearchMetrics):
    def __init__(self, sim, *, interval=50):
        super().__init__(sim, detailed=True, interval=interval)
        self.active = defaultdict(Counter)
        self.pending = Counter()
        self.pending_target = {}

    def before(self):
        # Same as the shared collector, without buffer stocks: the V3 protocol
        # exposes no reservation item view.
        sim = self.sim
        self.begin = sim.tick
        self.states = {
            m: "down" if s["down"] else s["status"].lower()
            for m, s in sim.machine_state.items()
        }
        self.stocks = {}
        self.agvs = {
            v: (a["job"] is not None, a["service"] is not None, tuple(a["cell"]))
            for v, a in sim.agvs.items()
        }

    def after(self, outcome):
        elapsed = self.sim.tick - self.begin
        before = self.agvs
        super().after(outcome)
        for v, a in self.sim.agvs.items():
            loaded, service, cell = before[v]
            if loaded:
                self.active[v]["active"] += elapsed
                if not service and tuple(a["cell"]) == cell:
                    self.active[v]["loaded_waiting"] += elapsed
                continue
            if a["job"] is not None:  # picked up this tick
                self.active[v]["active"] += self.pending.pop(v, 0) + elapsed
                continue
            target = a["target"]
            if target != self.pending_target.get(v):
                self.active[v]["idle"] += self.pending.pop(v, 0)
                self.pending_target[v] = target
            if target is not None and (service or tuple(a["cell"]) != cell):
                self.pending[v] += elapsed
            else:
                self.active[v]["idle"] += elapsed

    def finish(self, original):
        # Own summary: the shared collector's finish needs reservation-protocol
        # counters that the V3 protocol does not keep.
        duration = max(self.sim.tick - self.started_tick, 1)
        states = {m: dict(c) for m, c in sorted(self.machine.items())}
        machines = {m: c.get("processing", 0) / duration for m, c in states.items()}
        agvs = {
            v: c.get("active", 0) / duration for v, c in sorted(self.active.items())
        }
        return {
            "utilisation": {
                "duration_ticks": duration,
                "machine": machines,
                "machine_mean": sum(machines.values()) / max(len(machines), 1),
                "machine_states": states,
                "agv": agvs,
                "agv_mean": sum(agvs.values()) / max(len(agvs), 1),
                "agv_loaded_waiting_mean": sum(
                    c.get("loaded_waiting", 0) / duration for c in self.active.values()
                )
                / max(len(self.active), 1),
            },
            "delivery_ticks": sorted(
                e["tick"] for e in self.events if e["kind"] == "delivery"
            ),
            "outages": [
                # The engine names the machine "machine" in outage events.
                {
                    "kind": e["kind"],
                    "tick": e["tick"],
                    "machine": e.get("machine", e.get("machine_id")),
                }
                for e in self.events
                if e["kind"] in ("breakdown", "repair")
            ],
        }
