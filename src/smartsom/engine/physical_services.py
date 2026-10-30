"""Shared physical storage and one-tick vehicle services; no policy protocol."""


class PhysicalServices:
    def __init__(self, core):
        self.core = core
        self.ports = {p.port_id: p for p in core.factory.ports}
        self.matrix = None
        for vehicle in core.agvs.values():
            vehicle.update(
                target=None,
                reservation=None,
                reservation_tick=None,
                reservation_version=0,
                service=None,
                feedback=None,
                empty_notified=False,
                arrived_at=0,
            )
        self.sources = tuple(
            owner
            for owner in (*core.storage, *core.machines)
            if self.ports_for(owner, "pickup")
            and core.roles.get(owner) not in ("machine_pre", "system_output")
            and (owner not in core.machines or owner not in core.post)
        )

    def ports_for(self, owner, operation):
        return tuple(
            p
            for p in self.ports.values()
            if any(
                self.core._target(b.target)[0] == owner and operation in b.operations
                for b in p.bindings
            )
        )

    def complete_travel(self):
        # Grid movement is committed by the physical kernel, one interval at a time.
        pass

    def destination_owners(self, job):
        core, row = self.core, self.core.jobs[job]
        steps = core.demands[row["demand"]].steps
        if row["step"] < len(steps):
            return tuple(
                core.pre.get(m, m)
                for m, machine in core.machines.items()
                if steps[row["step"]].operation_type in machine.operation_types
            )
        return tuple(
            owner for owner in core.storage if core.roles[owner] == "system_output"
        )

    def ready(self, owner):
        core = self.core
        bound = {
            a["service"]["job"]
            for a in core.agvs.values()
            if a["service"] and a["service"]["kind"] == "pickup"
        }
        if owner in core.storage:
            if core.roles[owner] in ("machine_pre", "system_output"):
                return ()
            return tuple(
                sorted(
                    j
                    for rows in core.storage[owner].values()
                    for j in rows
                    if j not in bound
                )
            )
        if owner in core.machines:
            state = core.machine_state[owner]
            if owner not in core.post and state["status"] == "BLOCKED":
                return (state["job"],) if state["job"] not in bound else ()
        return ()

    def source_supply(self, owner):
        core = self.core
        supply = len(self.ready(owner))
        for machine, post in core.post.items():
            state = core.machine_state[machine]
            if (
                post == owner
                and state["status"] == "PROCESSING"
                and core._free(post) is not None
            ):
                supply += 1
        return supply

    def reserved(self, owner):
        return sum(
            a["job"] is None
            and a["target"] is not None
            and a["target"]["owner"] == owner
            for a in self.core.agvs.values()
        )

    def start_service(self, vehicle, kind, job, owner, slot):
        state = self.core.agvs[vehicle]
        state["service"] = {
            "kind": kind,
            "job": job,
            "owner": owner,
            "slot": slot,
            "started_at": self.core.tick,
            "remaining": 1,
        }
        self.core.metrics[f"{kind}_services"] += 1
        self.core._emit(f"{kind}_started", agv=vehicle, job=job, owner=owner, slot=slot)

    def complete_services(self):
        for vehicle, state in self.core.agvs.items():
            service = state["service"]
            if not service:
                continue
            service["remaining"] -= 1
            if service["remaining"]:
                continue
            transfer = tuple(service[k] for k in ("kind", "job", "owner", "slot"))
            self.core._transfer(vehicle, transfer)
            state["service"] = None
            state["target"] = None
            state["empty_notified"] = False
            state["feedback"] = f"{service['kind']}_completed"
            self.core._emit(
                f"{service['kind']}_completed", agv=vehicle, job=service["job"]
            )
