"""Request-scoped read-only queries over current public simulator information.

Queries are valid only during synchronous observation/rule construction. Queried
rows are immutable snapshots; retaining an unqueried live view is an error. This
avoids copying the whole facility for every local agent. No query exposes future
arrivals/outages, sampled durations, private quality draws, or mutation methods.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from smartsom.domain.public_facts import freeze
from smartsom.engine.public_facts import public_facts


class _Scope:
    """Shared lifetime guard without retaining the request's fact graph."""

    __slots__ = ("active",)

    def __init__(self):
        self.active = True

    def check(self):
        if not self.active:
            raise RuntimeError(
                "public queries are valid only during observation construction"
            )


class _Table(Mapping):
    def __init__(self, keys, getter, active):
        self._keys, self._getter, self._active = keys, getter, active
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._active()
            if key not in self._keys:
                raise KeyError(key)
            self._cache[key] = self._getter(key)
        return self._cache[key]

    def __contains__(self, key):
        self._active()
        return key in self._keys

    def __iter__(self):
        self._active()
        return iter(sorted(self._keys) if isinstance(self._keys, set) else self._keys)

    def __len__(self):
        self._active()
        return len(self._keys)


@dataclass(frozen=True, slots=True)
class _Geometry:
    grid: object
    operation_types: tuple


class _Routes:
    def __init__(self, scope, routes):
        self._scope, self._routes = scope, routes
        self.ports = MappingProxyType(
            {k: frozenset(v) for k, v in routes.ports.items()}
        )

    def to_owner(self, cell, owner):
        self._scope.check()
        return self._routes.to_owner(cell, owner)

    def local_roads(self, center, radius=2):
        self._scope.check()
        return self._routes.local_roads(center, radius)


class DecisionFacts(Mapping):
    def __init__(self, protocol, decision):
        core = protocol.core
        self._protocol, self._scope = protocol, _Scope()
        self._snapshot = None
        # Component-local derived values live only for this decision callback.
        # They cannot mutate physical state and are never reused at the next tick
        # or the next same-tick decision.
        self.component_cache = {}
        self.decision, self.tick = decision, core.tick
        self.service_jobs = frozenset(
            v["service"]["job"] for v in core.agvs.values() if v["service"]
        )
        self.scales = protocol.time_scale, protocol.distance_scale, protocol.count_scale
        self.factory = _Geometry(core.factory.grid, tuple(core.factory.operation_types))
        self.station = MappingProxyType(protocol.station)
        self.machines = MappingProxyType(core.machines)
        self.pre, self.post = MappingProxyType(core.pre), MappingProxyType(core.post)
        self.roles = MappingProxyType(core.roles)
        self.capacity = MappingProxyType(
            {k: MappingProxyType(v) for k, v in core.capacity.items()}
        )
        self.completed = frozenset(core.completed)
        self.idle_since = MappingProxyType(dict(protocol.idle_since))
        self.routes = _Routes(self._scope, protocol.routes)
        self.jobs = _Table(
            core.jobs,
            lambda j: MappingProxyType(
                {
                    k: core.jobs[j][k]
                    for k in (
                        "demand",
                        "location",
                        "slot",
                        "step",
                        "since",
                        "risk",
                        "quality",
                    )
                }
            ),
            self._scope.check,
        )
        self.demands = _Table(
            core.released, lambda j: core.demands[j], self._scope.check
        )

        def vehicle(v):
            row = core.agvs[v]
            return MappingProxyType(
                {
                    "cell": tuple(row["cell"]),
                    "job": row["job"],
                    "target": freeze(row["target"]),
                    "service": freeze(row["service"]),
                    "reservation_tick": row["reservation_tick"],
                    "arrived_at": row["arrived_at"],
                    "task_ticks": protocol.vehicle_task_ticks[v],
                }
            )

        self.agvs = _Table(core.agvs, vehicle, self._scope.check)
        self.machine_state = _Table(
            core.machine_state,
            lambda m: MappingProxyType(
                {
                    k: core.machine_state[m][k]
                    for k in ("job", "status", "down", "elapsed", "nominal", "mode")
                }
            ),
            self._scope.check,
        )
        self.storage = _Table(
            core.storage, lambda o: freeze(core.storage[o]), self._scope.check
        )

    def _check(self):
        self._scope.check()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._scope.active = False

    def snapshot(self):
        """Materialize the immutable full public view once for this decision."""
        if self._snapshot is None:
            self._check()
            self._snapshot = public_facts(self._protocol.core, topology=True)
        return self._snapshot

    def __getitem__(self, key):
        if key == "tick":
            return self.tick
        return self.snapshot()[key]

    def __iter__(self):
        return iter(self.snapshot())

    def __len__(self):
        return len(self.snapshot())

    def inventory(self, owner):
        self._check()
        return self._protocol.items(owner)

    def commitments(self, owner):
        self._check()
        return self._protocol.commitments(owner)

    def ready(self, owner):
        self._check()
        return self._protocol.ready(owner)

    def quota(self, owner):
        self._check()
        return self._protocol.quota(owner)

    def nominal(self, job, machine, mode="normal"):
        self._check()
        return self._protocol.nominal(job, machine, mode)

    def remaining_work(self, job):
        self._check()
        return self._protocol.remaining_work(job)

    def current_standard_work(self, machine):
        self._check()
        return self._protocol.current_standard_work(machine)

    def post_credit_components(self, owner):
        self._check()
        return self._protocol.post_credit_components(owner)

    def station_items(self, machine):
        self._check()
        return self._protocol.station_items(machine)

    def machine_choices(self, machine):
        self._check()
        return tuple(self._protocol.core.machine_choices(machine))

    def free_drop_count(self, owner):
        self._check()
        locked = {}
        for state in self.agvs.values():
            service = state["service"]
            if service and service["kind"] == "drop_off" and service["owner"] == owner:
                locked[service["slot"]] = locked.get(service["slot"], 0) + 1
        capacities = self.capacity[owner]
        if any(cap is None for cap in capacities.values()):
            return None
        return sum(
            max(0, cap - len(self.storage[owner][slot]) - locked.get(slot, 0))
            for slot, cap in capacities.items()
        )
