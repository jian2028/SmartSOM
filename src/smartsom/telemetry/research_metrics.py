"""Read-only reservation research telemetry, independent of Torch and policy inputs.

Summary mode is bounded by orders/resources, detailed mode adds sampled cumulative
counts and event records. Nothing here changes simulation state or consumes RNG.
"""

import hashlib
import json
import math
import time
from collections import Counter, defaultdict

from smartsom.config.codec import primitive

SCHEMA = "smartsom.research-metrics/v1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(row):
    """Derive metrics from historical or new rows without inventing absent data."""
    n, done, failed, tick = (
        row.get(k) for k in ("total_orders", "completed", "failed", "tick")
    )
    if any(v is None for v in (n, done, failed, tick)):
        return {"status": "not_recorded"}
    if any(type(v) is not int for v in (n, done, failed, tick)):
        raise ValueError("order counts and physical ticks must be integers")
    if not (0 <= failed <= done <= n and n > 0 and tick >= 0):
        raise ValueError("invalid order counts or observation duration")
    complete = done == n and not row.get("wall_clock_fragment", False)
    return {
        "delivery_fraction": done / n,
        "drained": complete,
        "unfinished": n - done,
        "qualified_coverage": (done - failed) / n,
        "conditional_passing_rate": ratio(done - failed, done),
        "final_passing_rate": (done - failed) / n if complete else None,
        "throughput_per_1000_ticks": ratio(1000 * done, tick),
        "qualified_throughput_per_1000_ticks": ratio(1000 * (done - failed), tick),
        "mean_tardiness": row.get("mean_tardiness") if complete else None,
        "joint_cost": row.get("joint_cost") if complete else None,
        "cost_status": (
            "observed"
            if row.get("mean_tardiness") is not None
            and row.get("joint_cost") is not None
            else "not_recorded"
        )
        if complete
        else "censored",
        "throughput_scope": "own_observation_horizon_not_common_window",
    }


class ResearchMetrics:
    def __init__(self, sim, *, detailed=False, interval=50):
        if type(interval) is not int or interval <= 0:
            raise ValueError("sample interval must be a positive integer")
        self.sim = sim
        self.detailed = detailed
        self.interval = interval
        self.started_tick = sim.tick
        self.machine = defaultdict(Counter)
        self.buffer = defaultdict(Counter)
        self.buffer_entities = defaultdict(Counter)
        self.vehicle = defaultdict(Counter)
        self.decisions = Counter()
        self.special = Counter()
        self.series = {}
        self.events = []
        self.entity_series = []
        self.last_entity_sample = sim.tick
        self.next_sample = sim.tick + interval
        self.deliveries = set(sim.shipment_times)
        self.seconds = 0.0
        if detailed:
            self.events.extend(
                dict(e)
                for e in sim.events
                if e["kind"] in ("breakdown", "repair", "demand_released")
            )
            self.series[sim.tick] = self.counts(sim.tick)
            self.entity_series.append(self.entity_snapshot())

    def decision(self, request, action):
        key = request.group + "." + request.stage
        self.decisions[key] += 1
        if action in (("STOP",), ("EMPTY_WAIT",)):
            self.special[key + "." + action[0]] += 1

    def counts(self, tick):
        sim = self.sim
        completed = {k for k, t in sim.shipment_times.items() if t <= tick}
        qualified = len(completed & sim._qualified_shipments)
        arrived = sum(d.release_at <= tick for d in sim.demands.values())
        return {
            "tick": tick,
            "completed": len(completed),
            "qualified": qualified,
            "failed": len(completed) - qualified,
            "unfinished": len(sim.demands) - len(completed),
            "arrived_backlog": arrived - len(completed),
            "future_unreleased": len(sim.demands) - arrived,
            "observed_overdue": sum(
                max(0, min(tick, sim.shipment_times.get(k, tick)) - d.due_at)
                for k, d in sim.demands.items()
            ),
        }

    def before(self):
        start = time.perf_counter()
        sim = self.sim
        self.begin = sim.tick
        self.states = {
            m: "down"
            if s["down"]
            else "idle"
            if s["status"] == "READY"
            else s["status"].lower()
            for m, s in sim.machine_state.items()
        }
        self.entity_stocks = {
            owner: sum(len(items) for items in slots.values())
            for owner, slots in sim.storage.items()
        }
        self.stocks = {
            m: {
                "pre": self.entity_stocks[sim.pre[m]],
                "post": self.entity_stocks[sim.post[m]],
            }
            for m in sim.machine_state
        }
        self.agvs = {
            v: (a["job"] is not None, a["service"] is not None, tuple(a["cell"]))
            for v, a in sim.agvs.items()
        }
        self.seconds += time.perf_counter() - start

    def after(self, outcome):
        start = time.perf_counter()
        sim = self.sim
        elapsed = sim.tick - self.begin
        if elapsed <= 0:
            raise ValueError("nonpositive observation interval")
        for m, state in self.states.items():
            self.machine[m][state] += elapsed
        for m, stocks in self.stocks.items():
            for phase, count in stocks.items():
                self.buffer[m][phase + "_resident_ticks"] += count * elapsed
        for owner, count in self.entity_stocks.items():
            self.buffer_entities[owner]["resident_ticks"] += count * elapsed
            self.buffer_entities[owner]["observed_ticks"] += elapsed
        for v, (loaded, service, cell) in self.agvs.items():
            self.vehicle[v]["loaded" if loaded else "empty"] += elapsed
            self.vehicle[v][
                "service_at_start" if service else "no_service_at_start"
            ] += elapsed
            self.vehicle[v][
                "moving" if cell != tuple(sim.agvs[v]["cell"]) else "stationary"
            ] += elapsed
        if not self.detailed:
            self.seconds += time.perf_counter() - start
            return
        if sim.tick - self.last_entity_sample >= self.interval:
            self.entity_series.append(self.entity_snapshot())
            self.last_entity_sample = sim.tick
        while self.next_sample <= sim.tick:
            self.series[self.next_sample] = self.counts(self.next_sample)
            self.next_sample += self.interval
        relevant = [
            dict(e)
            for e in outcome["events"]
            if e["kind"] in ("breakdown", "repair", "demand_released")
        ]
        self.events.extend(relevant)
        for event in relevant:
            self.series[event["tick"]] = self.counts(event["tick"])
        for key in sorted(set(sim.shipment_times) - self.deliveries):
            tick = sim.shipment_times[key]
            self.events.append(
                {
                    "kind": "delivery",
                    "tick": tick,
                    "order_id": key,
                    "qualified": key in sim._qualified_shipments,
                }
            )
            self.series[tick] = self.counts(tick)
        self.deliveries.update(sim.shipment_times)
        self.seconds += time.perf_counter() - start

    def entity_snapshot(self):
        return {
            "tick": self.sim.tick,
            "machines": {k: dict(self.machine[k]) for k in self.sim.machine_state},
            "vehicles": {k: dict(self.vehicle[k]) for k in self.sim.agvs},
            "buffers": {k: dict(self.buffer_entities[k]) for k in self.sim.storage},
        }

    def finish(self, original):
        sim = self.sim
        duration = sim.tick - self.started_tick
        machines = len(sim.machine_state)
        totals = {
            s: sim.protocol.metrics.get("machine_" + s + "_ticks", 0)
            for s in ("processing", "idle", "blocked", "down")
        }
        if sum(totals.values()) != machines * duration:
            raise ValueError("machine duration conservation failed")
        orders = []
        tardiness, flows = [], []
        on_time = 0
        for key, demand in sorted(sim.demands.items()):
            end = sim.shipment_times.get(key)
            if end is not None:
                tardiness.append(max(0, end - demand.due_at))
                flows.append(end - demand.release_at)
                on_time += end <= demand.due_at
            if self.detailed:
                orders.append(
                    {
                        "order_id": key,
                        "arrival_tick": demand.release_at,
                        "due_tick": demand.due_at,
                        "completion_tick": end,
                        "final_pass": key in sim._qualified_shipments
                        if end is not None
                        else None,
                        "status": "observed" if end is not None else "censored",
                    }
                )
        complete = len(flows) == len(sim.demands)
        # Nearest-rank tails, conditional on completed orders unless all drained.
        tails = {
            f"tardiness_p{p}": sorted(tardiness)[
                math.ceil(len(tardiness) * p / 100) - 1
            ]
            if tardiness
            else None
            for p in (50, 95, 99)
        }
        summary = {
            **aggregate(original),
            **tails,
            "distribution_scope": "all_orders" if complete else "completed_orders_only",
            "distribution_count": len(flows),
            "mean_flow_time": ratio(sum(flows), len(flows)),
            "on_time_coverage": on_time / len(sim.demands),
            "conditional_on_time_rate": ratio(on_time, len(flows)),
            "machine_state_ticks": totals,
            "machine_state_fractions": {
                s: ratio(v, machines * duration) for s, v in totals.items()
            },
            "available_processing_fraction": ratio(
                totals["processing"], machines * duration - totals["down"]
            ),
            "decision_counts": dict(self.decisions),
            "special_action_counts": dict(self.special),
        }
        result = {
            "schema": SCHEMA,
            "level": "detailed" if self.detailed else "summary",
            "start_tick": self.started_tick,
            "end_tick": sim.tick,
            "summary": summary,
            "machines": dict(self.machine),
            "vehicles": dict(self.vehicle),
            "buffer_entities": dict(self.buffer_entities),
            "missing": {
                "car": "not_recorded: requires a frozen learning target",
                "social_influence_gain": "not_recorded: requires a matched Base comparison",
                "task_volatility": "not_recorded: task graph change definition required",
                "team_complexity": "not_recorded: interaction index and beta required",
                "team_heterogeneity": "not_recorded: agent population and type definition required",
                "norm_stability": "not_applicable: no organizational norm module",
                "robustness": "not_recorded: requires paired common windows",
                "adaptability": "not_recorded: requires a frozen recovery protocol",
                "energy": "not_applicable: energy physics unsupported",
                "exact_internal_agv_service_ticks": "not_recorded",
            },
        }
        if self.detailed:
            if not self.entity_series or self.entity_series[-1]["tick"] != sim.tick:
                self.entity_series.append(self.entity_snapshot())
            self.series[sim.tick] = self.counts(sim.tick)
            for state, value in totals.items():
                if sum(m[state] for m in self.machine.values()) != value:
                    raise ValueError("per-machine durations differ from protocol")
            scenario = primitive(sim.scenario)
            exogenous = {
                k: v for k, v in scenario.items() if k not in ("outages", "tick_limit")
            }
            result.update(
                scenario_counts={
                    "orders": len(sim.demands),
                    "operation_nodes": sum(len(d.steps) for d in sim.demands.values()),
                    "precedence_edges": sum(
                        len(d.steps) - 1 for d in sim.demands.values()
                    ),
                    "machines": machines,
                    "vehicles": len(sim.agvs),
                },
                entity_series=self.entity_series,
                sample_interval_ticks=self.interval,
                orders=orders,
                machines=dict(self.machine),
                buffers=dict(self.buffer),
                vehicles=dict(self.vehicle),
                events=sorted(self.events, key=lambda e: e["tick"]),
                series=[self.series[t] for t in sorted(self.series)],
                fingerprints={
                    "exogenous": fingerprint(exogenous),
                    "orders": fingerprint(sim.scenario.demands),
                    "outages": fingerprint(sim.scenario.outages),
                },
                outage_realization=primitive(sim.scenario.outages),
                observer_seconds=self.seconds,
                definitions={
                    "resources": "pre-advance state times physical elapsed; down takes precedence; ready awaiting assignment counts as idle",
                    "buffers": "resident item*tick, excludes reservations",
                    "vehicles": "loaded/empty, moving/stationary, service_at_start/no_service_at_start are separate overlapping partitions; service is boundary observed",
                    "series": "end-of-tick cumulative counts; previous sample points exclude later deliveries",
                    "observer_seconds": "before/after sampling only; excludes decisions, finalization and serialization",
                },
            )
        return result
