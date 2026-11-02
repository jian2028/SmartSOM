"""Passive dispatcher transit accounting from ordered committed physics events.

The current kernel moves one grid edge in one tick. A move event is stamped at
the start of that interval, so its arrival is at event.tick + 1. No policy, RNG,
reward, simulation execution or per-tick route search belongs in this observer.
"""

import math
from collections import Counter

SCHEMA = "smartsom.trip-transit/v1"
KINDS = ("all", "empty", "loaded")
COUNTS = ("accepted", "completed", "cancelled", "retargeted", "censored")
CASE_FIELDS = (
    *COUNTS,
    "completed_ratio_count",
    "zero_distance",
    "unreachable",
    "censored_elapsed_ticks",
    "censored_ideal_ticks",
    "censored_ratio_count",
    "censored_ratio_lower_bound_sum",
)


def summary_fields(groups):
    """Flat case metrics; later repeat/world/seed means retain their own weights."""
    result = {}
    for kind, row in groups.items():
        prefix = "agv_transit_" + ("" if kind == "all" else kind + "_")
        result.update({prefix + key: row[key] for key in CASE_FIELDS})
        result.update(
            {
                prefix + "ratio_mean": row["completed_ratio_mean"],
                prefix + "elapsed_over_ideal": row["completed_elapsed_over_ideal"],
            }
        )
    return result


class TripTransitMetrics:
    def __init__(self, routes, vehicles, goals, *, sink=None):
        self.routes, self.goals, self.sink = routes, goals, sink
        self.goal_sets = {key: frozenset(cells) for key, cells in goals.items()}
        self.cells = {v: tuple(a["cell"]) for v, a in vehicles.items()}
        self.jobs = {v: a["job"] for v, a in vehicles.items()}
        self.sequence = Counter()
        self.active = {}
        self.totals = {kind: Counter() for kind in KINDS}

    def observe(self, events):
        for event in events:
            kind, vehicle = event["kind"], event.get("agv")
            if kind == "reservation_accepted":
                self._accept(vehicle, event["owner"], event["tick"])
            elif kind == "reservation_cancelled":
                self._close(vehicle, event["tick"], "cancelled", event.get("reason"))
            elif kind == "move":
                self.cells[vehicle] = tuple(event["after"])
                trip = self.active.get(vehicle)
                if trip is not None:
                    trip["move_ticks"] += 1
                    if (
                        self.cells[vehicle]
                        in self.goal_sets[trip["target_owner"], trip["operation"]]
                    ):
                        self._close(vehicle, event["tick"] + 1, "completed")
            elif kind == "pickup":
                self.jobs[vehicle] = event["job"]
            elif kind == "drop_off":
                self.jobs[vehicle] = None

    def _accept(self, vehicle, owner, tick):
        previous = self.active.get(vehicle)
        if previous is not None:
            if previous["target_owner"] == owner:
                # Traffic replans do not create reservations. Repeated evidence
                # for an unchanged active destination must not reset its clock.
                return
            self._close(vehicle, tick, "retargeted", "target_changed")
        loaded = self.jobs[vehicle] is not None
        kind, operation = ("loaded", "drop_off") if loaded else ("empty", "pickup")
        goals = self.goals[(owner, operation)]
        distance = self.routes.distances(goals).get(self.cells[vehicle], math.inf)
        ideal = distance if math.isfinite(distance) else None
        self.sequence[vehicle] += 1
        self.active[vehicle] = {
            "schema": SCHEMA,
            "agv_id": vehicle,
            "trip_sequence": self.sequence[vehicle],
            "kind": kind,
            "job_id_at_accept": self.jobs[vehicle],
            "target_owner": owner,
            "operation": operation,
            "goal_cells": goals,
            "start_cell": self.cells[vehicle],
            "accepted_tick": tick,
            "ideal_transit_ticks": ideal,
            "move_ticks": 0,
            "ratio_eligibility": "unreachable"
            if ideal is None
            else "zero_distance"
            if ideal == 0
            else "positive_ideal",
        }
        for scope in ("all", kind):
            self.totals[scope]["accepted"] += 1
        if self.cells[vehicle] in goals:
            self._close(vehicle, tick, "completed")

    def _close(self, vehicle, tick, status, reason=None):
        trip = self.active.pop(vehicle, None)
        if trip is None:
            return
        elapsed = tick - trip["accepted_tick"]
        ideal = trip["ideal_transit_ticks"]
        valid = ideal is not None and ideal > 0
        ratio = elapsed / ideal if valid else None
        for scope in ("all", trip["kind"]):
            totals = self.totals[scope]
            totals[status] += 1
            if not valid:
                totals[trip["ratio_eligibility"]] += 1
            if status == "completed" and valid:
                totals["completed_ratio_count"] += 1
                totals["completed_ratio_sum"] += ratio
                totals["completed_elapsed_ticks"] += elapsed
                totals["completed_ideal_ticks"] += ideal
            elif status == "censored":
                totals["censored_elapsed_ticks"] += elapsed
                if valid:
                    totals["censored_ratio_count"] += 1
                    totals["censored_ratio_lower_bound_sum"] += ratio
                    totals["censored_ideal_ticks"] += ideal
        if self.sink is not None:
            self.sink(
                {
                    **trip,
                    "end_tick": tick,
                    "end_cell": self.cells[vehicle],
                    "status": status,
                    "reason": reason,
                    "elapsed_ticks": elapsed,
                    "stationary_ticks": elapsed - trip["move_ticks"],
                    "completed_ratio": ratio if status == "completed" else None,
                    "elapsed_over_ideal_lower_bound": ratio
                    if status == "censored"
                    else None,
                }
            )

    def finish(self, tick):
        for vehicle in tuple(self.active):
            self._close(vehicle, tick, "censored", "episode_ended_before_arrival")
        groups = {}
        for kind, totals in self.totals.items():
            count = totals["completed_ratio_count"]
            ideal = totals["completed_ideal_ticks"]
            groups[kind] = {
                **{
                    key: totals[key]
                    for key in (
                        *COUNTS,
                        "zero_distance",
                        "unreachable",
                        "completed_ratio_count",
                        "completed_ratio_sum",
                        "completed_elapsed_ticks",
                        "completed_ideal_ticks",
                        "censored_elapsed_ticks",
                        "censored_ideal_ticks",
                        "censored_ratio_count",
                        "censored_ratio_lower_bound_sum",
                    )
                },
                "completed_ratio_mean": totals["completed_ratio_sum"] / count
                if count
                else None,
                "completed_elapsed_over_ideal": totals["completed_elapsed_ticks"]
                / ideal
                if ideal
                else None,
            }
        return {
            "schema": SCHEMA,
            "scope": "dispatcher_accepted_to_first_legal_port_arrival",
            "time_unit": "physical_tick",
            "ideal": "static_four_neighbor_shortest_edges_at_one_edge_per_tick",
            "groups": groups,
        }
