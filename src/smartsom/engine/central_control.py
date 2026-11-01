"""Direct task assignments composed with the shared physical services."""

import json
import math
from dataclasses import dataclass

from smartsom.domain.production import MachineCommand


@dataclass(frozen=True, slots=True)
class CentralTask:
    identity: str
    kind: str
    job: str | None = None
    machine: str | None = None
    mode: str | None = None
    vehicle: str | None = None
    destination: str | None = None


def task(kind, **values):
    identity = json.dumps([kind, *values.values()], separators=(",", ":"))
    return CentralTask(identity, kind, **values)


ADVANCE = task("advance")


@dataclass(frozen=True, slots=True)
class CentralDecision:
    tick: int
    candidates: tuple[CentralTask, ...]
    inputs: object
    group: str = "centralized"
    owner: str = "factory"
    stage: str = "assign"


class CentralControl:
    def __init__(self, core):
        self.core, self.protocol = core, core.protocol
        self.assignments = {}
        self.protocol.central = self

    def decisions(self):
        from smartsom.engine.public_facts import public_facts

        self.core.events = []
        self.reconcile()
        while True:
            request = CentralDecision(
                self.core.tick, self.candidates(), public_facts(self.core)
            )
            if self.protocol.observation_transform is not None:
                request = self.protocol.observation_transform(request.inputs, request)
            if not self.choose((yield request)):
                break
        self.start_services()

    def destinations_reserved(self, owner):
        return tuple(
            v
            for v, assignment in self.assignments.items()
            if assignment.destination == owner
        )

    def cancel(self, vehicle):
        self.assignments.pop(vehicle, None)

    def reconcile(self):
        for vehicle, assignment in tuple(self.assignments.items()):
            state = self.core.agvs[vehicle]
            if not self.protocol.eligible_owner(assignment.destination):
                self.protocol.cancel(vehicle, "central_destination_unavailable")
                self.assignments.pop(vehicle, None)
                continue
            if state["job"] == assignment.job and state["target"] is None:
                self._target(vehicle, assignment.destination)
            elif state["job"] is None and (
                self.core.jobs[assignment.job]["location"] == assignment.destination
            ):
                self.assignments.pop(vehicle, None)

    def before_services(self):
        # Availability at this timestamp belongs to physics, not an observation
        # of the future fault calendar. Cancel before a pending pickup transfers.
        for vehicle, assignment in tuple(self.assignments.items()):
            machine = self.protocol.station.get(assignment.destination)
            if machine and self.protocol.down_at(machine, self.core.tick):
                self.protocol.cancel(vehicle, "central_destination_breakdown")

    def _target(self, vehicle, owner):
        state = self.core.agvs[vehicle]
        ports = self.protocol.ports_for(owner, "drop_off" if state["job"] else "pickup")
        port = min(
            ports,
            key=lambda p: (
                self.protocol.routes.distances(((p.cell.x, p.cell.y),)).get(
                    tuple(state["cell"]), math.inf
                ),
                p.port_id,
            ),
        )
        state.update(
            target={"owner": owner, "port": port.port_id},
            reservation=owner,
            reservation_tick=self.core.tick,
            arrived_at=self.core.tick,
        )

    def candidates(self):
        core, protocol = self.core, self.protocol
        reserved_jobs = {a.job for a in self.assignments.values()}
        result = []
        for machine in core.machines:
            for job, mode in core.machine_choices(machine):
                if job not in reserved_jobs:
                    result.append(task("process", job=job, machine=machine, mode=mode))
        for vehicle, state in core.agvs.items():
            if vehicle in self.assignments or state["service"] or state["target"]:
                continue
            if state["job"]:
                jobs = ((state["job"], None),)
            else:
                jobs = tuple(
                    (j, owner)
                    for owner in protocol.sources
                    if protocol.eligible_owner(owner)
                    for j in protocol.ready(owner)
                    if j not in reserved_jobs
                )
            for job, source in jobs:
                if source and not math.isfinite(
                    protocol.routes.to_owner(state["cell"], source)
                ):
                    continue
                for destination in protocol.destination_owners(job):
                    if (
                        protocol.eligible_owner(destination)
                        and protocol.quota(destination) > 0
                    ):
                        if math.isfinite(
                            protocol.routes.to_owner(state["cell"], destination)
                        ):
                            result.append(
                                task(
                                    "transport",
                                    job=job,
                                    vehicle=vehicle,
                                    destination=destination,
                                )
                            )
        return (*result, ADVANCE)

    def choose(self, identity):
        candidate = next((c for c in self.candidates() if c.identity == identity), None)
        if candidate is None:
            raise ValueError("illegal or stale centralized semantic task")
        if candidate.kind == "advance":
            return False
        if candidate.kind == "process":
            self.core._start(
                candidate.machine, MachineCommand(candidate.job, candidate.mode)
            )
        else:
            state = self.core.agvs[candidate.vehicle]
            self.assignments[candidate.vehicle] = candidate
            source = self.core.jobs[candidate.job]["location"]
            self._target(
                candidate.vehicle, candidate.destination if state["job"] else source
            )
            self.core._emit(
                "task_assigned",
                agv=candidate.vehicle,
                job=candidate.job,
                destination=candidate.destination,
            )
        self.protocol.check_reservations()
        return True

    def start_services(self):
        self.reconcile()
        for vehicle, assignment in sorted(self.assignments.items()):
            state = self.core.agvs[vehicle]
            if state["service"] or state["target"] is None:
                continue
            owner = state["target"]["owner"]
            if tuple(state["cell"]) not in self.protocol.routes.ports[owner]:
                continue
            if state["job"]:
                slot = self.protocol.free_drop(owner)
                if slot is None:
                    continue
                kind = "drop_off"
            else:
                if assignment.job not in self.protocol.ready(owner):
                    continue
                slot = self.core.jobs[assignment.job]["slot"]
                kind = "pickup"
            self.protocol.start_service(vehicle, kind, assignment.job, owner, slot)

    def after_services(self):
        self.reconcile()
