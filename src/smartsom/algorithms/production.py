"""Rules over detached public decisions; policies never receive the simulator."""

import random
from collections import deque

from smartsom.domain.factory_design import occupied_cells
from smartsom.domain.production import AGV_ACTIONS, MOVES, JointCommand, MachineCommand


class ScriptedProductionPolicy:
    """One explicit semantic joint command per committed simulation tick."""

    def __init__(self, commands):
        self.commands = tuple(commands)
        self.position = 0
        self.scores = {}

    def rank(self, view):
        if self.position >= len(self.commands):
            raise ValueError(
                "scripted commands exhausted before the simulation finished"
            )
        return dict(self.commands[self.position].rankings)

    def act(self, view):
        command = self.commands[self.position]
        self.position += 1
        return command


class GreedyProductionPolicy:
    def __init__(self, factory, seed=0, *, rule="greedy", quality_mode=None):
        self.factory = factory
        self.rule = rule
        self.quality_mode = quality_mode
        self.rng = random.Random(seed)
        # With one capacity-one mover and no intermediate storage, two jobs
        # needing each other's occupied machines cannot be swapped. Conservative
        # release is a rule-policy choice; the core still permits concurrent WIP.
        self.serial_release = (
            len(factory.agvs) == 1
            and not factory.inspection_stations
            and all(
                b.role in ("system_input", "system_output") for b in factory.buffers
            )
        )
        self.boundary_buffers = {
            b.buffer_id
            for b in factory.buffers
            if b.role in ("system_input", "system_output")
        }
        self.buffer_roles = {b.buffer_id: b.role for b in factory.buffers}
        self.station_ids = {
            s.inspection_station_id for s in factory.inspection_stations
        }
        self.ports = {p.port_id: (p.cell.x, p.cell.y) for p in factory.ports}
        self.solids = {(c.x, c.y) for c in factory.grid.blocked_cells}
        for group in (
            factory.machines,
            factory.buffers,
            factory.inspection_stations,
            factory.scrap_bins,
            factory.chargers,
        ):
            for resource in group:
                self.solids.update(
                    (c.x, c.y) for c in occupied_cells(resource.footprint)
                )

    def rank(self, view):
        if self.rule in ("spt", "first_feasible"):
            self.scores = {
                owner: {
                    job: -min(
                        view["jobs"][job]["machine_nominal_ticks"].values(), default=0
                    )
                    if self.rule == "spt"
                    else 0
                    for job in jobs
                }
                for owner, jobs in view["rankings"].items()
            }
            return {
                owner: tuple(sorted(jobs, key=lambda j: (-self.scores[owner][j], j)))
                for owner, jobs in view["rankings"].items()
            }
        scale = 1 + max((j["priority"] for j in view["jobs"].values()), default=1)
        self.scores = {
            owner: {
                j: -view["jobs"][j]["due_at"] + view["jobs"][j]["priority"] / scale
                for j in jobs
            }
            for owner, jobs in view["rankings"].items()
        }
        return {
            owner: tuple(
                sorted(
                    jobs,
                    key=lambda j: (
                        view["jobs"][j]["due_at"],
                        -view["jobs"][j]["priority"],
                        j,
                    ),
                )
            )
            for owner, jobs in view["rankings"].items()
        }

    def path(self, start, goal, blocked=()):
        blocked = set(blocked) | self.solids
        queue = deque([(tuple(start), ())])
        seen = {tuple(start)}
        while queue:
            cell, path = queue.popleft()
            if cell == goal:
                return path
            for action, (dx, dy) in MOVES.items():
                nxt = cell[0] + dx, cell[1] + dy
                if (
                    0 <= nxt[0] < self.factory.grid.width
                    and 0 <= nxt[1] < self.factory.grid.height
                    and nxt not in seen
                    and nxt not in blocked
                ):
                    seen.add(nxt)
                    queue.append((nxt, (*path, action)))
        return None

    def act(self, view):
        machines = []
        claimed = set()
        for machine, choices in view["machine_choices"].items():
            if self.quality_mode is not None:
                choices = [
                    choice for choice in choices if choice[1] == self.quality_mode
                ]
            if choices:

                def key(choice):
                    if self.rule != "spt":
                        return (choice[1] != "normal", choice)
                    mode = next(
                        m
                        for m in next(
                            m for m in self.factory.machines if m.machine_id == machine
                        ).quality_modes
                        if m.quality_mode_id == choice[1]
                    )
                    return (
                        view["jobs"][choice[0]]["machine_nominal_ticks"][machine]
                        * float(mode.time_scale),
                        choice,
                    )

                choice = min(choices, key=key)
                machines.append((machine, MachineCommand(*choice)))
                claimed.add(choice[0])
        quality = [
            (key, "START") for key, ready in view["inspection_choices"].items() if ready
        ]
        inspection_locked = {key for key, _ in quality}
        agvs = []
        occupied = {tuple(v["cell"]) for v in view["agvs"].values()}
        reserved = set()
        # Loaded vehicles plan first. Reservations are local to this policy's
        # joint proposal; they do not change core arbitration or public masks.
        vehicles = sorted(
            view["agvs"].items(), key=lambda row: (row[1]["job"] is None, row[0])
        )
        for key, vehicle in vehicles:
            cell = tuple(vehicle["cell"])
            blocked = (occupied - {cell}) | reserved
            options = []
            for port, transfer in view["interactions"][key].items():
                if (
                    transfer is None
                    or transfer[1] in claimed
                    or transfer[2] in inspection_locked
                ):
                    continue
                if transfer[0] == "pickup":
                    # Avoid taking a new input while a blocked machine needs this
                    # capacity-one vehicle to remove its previous job.
                    source = next(
                        (b for b in self.factory.buffers if b.buffer_id == transfer[2]),
                        None,
                    )
                    if source and source.role == "system_input":
                        if self.serial_release and any(
                            job["location"] not in self.boundary_buffers
                            for job in view["jobs"].values()
                        ):
                            continue
                        kind = view["jobs"][transfer[1]]["next_operation"]
                        if not any(
                            kind in m.operation_types
                            and view["machines"][m.machine_id]["job"] is None
                            and not view["machines"][m.machine_id]["down"]
                            for m in self.factory.machines
                        ):
                            continue
                if self.ports[port] in blocked:
                    continue
                path = self.path(cell, self.ports[port], blocked)
                if path is not None:
                    owner = transfer[2]
                    priority = 0
                    if transfer[0] == "drop":
                        if self.buffer_roles.get(owner) == "storage":
                            priority = 2
                        elif (
                            owner in self.station_ids
                            and view["jobs"][transfer[1]]["next_operation"] is not None
                        ):
                            priority = 1
                    elif self.buffer_roles.get(owner) == "system_input":
                        priority = 1
                    options.append((priority, len(path), port, path, transfer))
            if options:
                _, _, _, path, transfer = min(options)
                action = path[0] if path else "INTERACT"
                claimed.add(transfer[1])
            else:
                action = "WAIT"
                # An idle vehicle must release a port so another can interact.
                # Park at the nearest reachable non-port cell; no teleportation.
                if len(vehicles) > 1 and cell in self.ports.values():
                    parking = []
                    for x in range(self.factory.grid.width):
                        for y in range(self.factory.grid.height):
                            target = (x, y)
                            if (
                                target in blocked
                                or target in self.solids
                                or target in self.ports.values()
                            ):
                                continue
                            path = self.path(cell, target, blocked)
                            if path:
                                parking.append((len(path), target, path))
                    if parking:
                        action = min(parking)[2][0]
            if action in MOVES:
                dx, dy = MOVES[action]
                reserved.add((cell[0] + dx, cell[1] + dy))
            else:
                reserved.add(cell)
            agvs.append((key, action))
        return JointCommand(
            tuple(agvs),
            tuple(machines),
            tuple(quality),
            tuple((k, tuple(v)) for k, v in view["rankings"].items()),
        )


class RandomProductionPolicy(GreedyProductionPolicy):
    def rank(self, view):
        self.scores = {}
        rows = {}
        for owner, jobs in view["rankings"].items():
            jobs = list(jobs)
            self.scores[owner] = {j: self.rng.random() for j in jobs}
            jobs.sort(key=lambda j: (-self.scores[owner][j], j))
            rows[owner] = tuple(jobs)
        return rows

    def act(self, view):
        agvs = tuple(
            (
                key,
                self.rng.choice(
                    [a for a, valid in zip(AGV_ACTIONS, mask, strict=True) if valid]
                ),
            )
            for key, mask in view["agv_masks"].items()
        )
        machines = tuple(
            (
                key,
                self.rng.choice(
                    [MachineCommand(), *(MachineCommand(*c) for c in choices)]
                ),
            )
            for key, choices in view["machine_choices"].items()
        )
        quality = tuple(
            (key, self.rng.choice(["START", "WAIT"]) if ready else "WAIT")
            for key, ready in view["inspection_choices"].items()
        )
        return JointCommand(
            agvs,
            machines,
            quality,
            tuple((k, tuple(v)) for k, v in view["rankings"].items()),
        )
