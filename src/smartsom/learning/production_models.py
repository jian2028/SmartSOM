"""Optional ragged candidate networks; masks never truncate a candidate pool."""

import json
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from smartsom.config.codec import primitive
from smartsom.config.extensions import ActorCriticSpec
from smartsom.config.policies import QNetwork
from smartsom.domain.factory_design import occupied_cells
from smartsom.learning.extensions import ObservationSpace
from smartsom.learning.torch_extensions import _Branch

CANDIDATE_WIDTH = 16
# has machine, idle, processing, ready/blocked, down, expected remaining, queue
MACHINE_FEATURES = 7
# Joint dispatch pair: is a pair, source-to-machine distance, earliest slack.
JOINT_FEATURES = 3
PREFIX_WIDTH = 32


class PublicEncoder:
    DIAGNOSTIC_CONTRACT = {
        "schema": "smartsom.public-encoder/v1",
        "actor_scope": "owner context plus public global resources and topology",
        "critic_only_fields": [],
    }

    def __init__(
        self,
        factory,
        projection,
        *,
        observation=None,
        provider="rllib.resource_ppo",
        role="machine",
        training=True,
        network=None,
    ):
        self.factory, self.projection = factory, dict(projection)
        self.role, self.training = role, training
        options = dict(network or {})
        self.supply_mask = bool(options.get("dispatch_supply_mask"))
        self.commit_mask = bool(options.get("dispatch_commit"))
        self.exclusive = bool(options.get("dispatch_tick_exclusive"))
        self.idle_fallback = self.exclusive and bool(
            options.get("dispatch_idle_fallback")
        )
        self._round = (None, set(), Counter(), Counter(), set(), Counter())
        self.machine_features = bool(options.get("target_machine_features"))
        self.joint = (
            self.exclusive
            and self.commit_mask
            and self.machine_features
            and bool(options.get("dispatch_joint_destination"))
        )
        # vehicle -> (source owner, destination machine) chosen while empty;
        # the expansion of the last encoded request maps choices to candidates.
        self.commitments, self._expansion = {}, (None, None)
        # Target buffer -> the machine it feeds or drains, and machine -> input buffer.
        self.feeds = {
            b.buffer_id: b.machine_id
            for b in factory.buffers
            if b.role in ("machine_pre", "machine_post") and b.machine_id
        }
        self.feeds.update({m.machine_id: m.machine_id for m in factory.machines})
        self.inputs = {
            b.machine_id: b.buffer_id
            for b in factory.buffers
            if b.role == "machine_pre" and b.machine_id
        }
        self.owners = sorted(
            {
                *(m.machine_id for m in factory.machines),
                *(a.agv_id for a in factory.agvs),
                *(b.buffer_id for b in factory.buffers),
                *(s.inspection_station_id for s in factory.inspection_stations),
                *(b.scrap_bin_id for b in factory.scrap_bins),
            }
        )
        self.port_ids = sorted(p.port_id for p in factory.ports)
        self.operation_types = sorted(factory.operation_types)
        self.candidate_width = (
            CANDIDATE_WIDTH
            + 2 * len(self.operation_types)
            + len(self.owners)
            + len(self.port_ids)
            + 2 * len(factory.machines)
            + (MACHINE_FEATURES if self.machine_features else 0)
            + (JOINT_FEATURES if self.joint else 0)
        )
        self.machine_ids = sorted(m.machine_id for m in factory.machines)
        self.port_cells = {p.port_id: (p.cell.x, p.cell.y) for p in factory.ports}
        self.buffer_cells = {
            b.buffer_id: [(c.x, c.y) for c in occupied_cells(b.footprint)]
            for b in factory.buffers
        }
        self.machine_operations = {
            m.machine_id: set(m.operation_types) for m in factory.machines
        }
        self.agv_ids = sorted(a.agv_id for a in factory.agvs)
        self.capacities = {key: 1 for key in (*self.machine_ids, *self.agv_ids)}
        for buffer in factory.buffers:
            if hasattr(buffer.storage, "slots"):
                self.capacities[buffer.buffer_id] = sum(
                    slot.capacity for slot in buffer.storage.slots
                )
            else:
                self.capacities[buffer.buffer_id] = buffer.storage.capacity
        for station in factory.inspection_stations:
            self.capacities[station.inspection_station_id] = sum(
                slot.capacity for slot in station.slots
            )
        for sink in factory.scrap_bins:
            self.capacities[sink.scrap_bin_id] = sink.capacity
        solids = {(c.x, c.y) for c in factory.grid.blocked_cells}
        for field in (
            "machines",
            "buffers",
            "inspection_stations",
            "scrap_bins",
            "chargers",
        ):
            for resource in getattr(factory, field):
                solids.update((c.x, c.y) for c in occupied_cells(resource.footprint))
        self.solids = solids
        self.map = [
            float((x, y) in solids)
            for y in range(factory.grid.height)
            for x in range(factory.grid.width)
        ]
        # header, owner identity, self target/reservation/port, local 5x5,
        # public global resources and static topology.
        self.context_size = (
            16
            + len(self.owners) * 3
            + len(self.port_ids)
            + 50
            + len(self.agv_ids) * 5
            + len(self.machine_ids) * 8
            + len(self.map)
            + len(self.owners) * (9 + len(self.operation_types))
        )

        self.central_private_end = 16 + len(self.owners) * 3 + len(self.port_ids) + 50
        self.base_context_size = self.context_size
        from smartsom.config.extensions import ExtensionRef, ExtensionSpec
        from smartsom.learning.extensions import ExtensionsRuntime, PublicObservation

        self.extension_role = None if role == "central" else role + "_policy"
        public = PublicObservation(
            None,
            (0.0,) * self.context_size,
            (("state", (0.0,) * self.context_size),),
            self.extension_role,
        )
        spec = (
            ExtensionSpec(
                observation=ExtensionRef.model_validate_json(
                    __import__("json").dumps(observation)
                )
            )
            if observation
            else None
        )
        self.extensions = ExtensionsRuntime(
            spec, provider, {self.extension_role: public.layout()}
        )
        self.context_size = self.extensions.spaces[self.extension_role].flat_size

    def state_dict(self):
        return self.extensions.state_dict()

    def load_state_dict(self, state):
        self.extensions.load_state_dict(state)

    def set_training(self, training):
        self.training = training
        for component in self.extensions.components.values():
            hook = getattr(component.instance, "set_training", None)
            if hook:
                hook(training)

    def semantic_features(self, job, observation, *, target=None, after_start=False):
        row = observation["jobs"].get(job, {})
        steps = row.get("remaining_steps", [])
        upcoming = steps[1:] if after_start else steps
        operation = upcoming[0]["operation_type"] if upcoming else None
        location = target.owner if target else row.get("location")
        port = target.port if target else None
        nominal = dict(upcoming[0]["machine_nominal_ticks"]) if upcoming else {}
        return [
            *[float(operation == kind) for kind in self.operation_types],
            *[
                sum(step["operation_type"] == kind for step in upcoming)
                / self.projection["count_scale"]
                for kind in self.operation_types
            ],
            *[float(location == owner) for owner in self.owners],
            *[float(port == key) for key in self.port_ids],
            *[
                float(operation in machine.operation_types)
                for machine in sorted(self.factory.machines, key=lambda m: m.machine_id)
            ],
            *[
                (
                    nominal.get(machine.machine_id, upcoming[0]["nominal_ticks"])
                    / self.projection["time_scale"]
                )
                if upcoming and operation in machine.operation_types
                else 0.0
                for machine in sorted(self.factory.machines, key=lambda m: m.machine_id)
            ],
        ]

    def job_prefix(self, job, observation):
        row = observation["jobs"][job]
        scale, count = self.projection["time_scale"], self.projection["count_scale"]
        risk = float(row.get("risk", -1))
        return [
            len(row["remaining_steps"]) / 10,
            (row["due_at"] - observation["tick"]) / scale,
            row["priority"] / 10,
            (observation["tick"] - row["since"]) / scale,
            row["attempt"] / 10,
            float(row["quality"] == "UNKNOWN"),
            float(row["quality"] == "PASS"),
            float(row["quality"] == "FAIL"),
            0.0,
            0.0,
            0.0,
            0.0,
            risk if risk >= 0 else 0.0,
            float(risk >= 0),
            len(row["remaining_steps"]) / count,
            1.0,
            *self.semantic_features(job, observation),
            *([0.0] * MACHINE_FEATURES if self.machine_features else []),
            *([0.0] * JOINT_FEATURES if self.joint else []),
        ]

    def target_machine(self, owner, observation):
        """Public state of the machine a target buffer feeds or drains."""
        machine = self.feeds.get(owner)
        if machine is None:
            return [0.0] * MACHINE_FEATURES
        state = observation["machines"][machine]
        queue = sum(
            row["location"] == self.inputs.get(machine)
            for row in observation["jobs"].values()
        )
        remaining = max(state["nominal"] - state["elapsed"], 0) if state["job"] else 0
        return [
            1.0,
            float(state["status"] == "IDLE"),
            float(state["status"] == "PROCESSING"),
            float(state["status"] in ("READY", "BLOCKED")),
            float(state["down"]),
            remaining / self.projection["time_scale"],
            queue / self.projection["count_scale"],
        ]

    def dispatch_round(self, request):
        """Choices this controller already made in the same dispatch boundary.

        All Dispatcher requests of one boundary share one public snapshot, so
        tick, stage and vehicle state identify the round even when several
        environments share this encoder.
        """
        view = request.observation
        key = (
            request.tick,
            request.stage,
            json.dumps(view["agvs"], sort_keys=True, default=str),
        )
        if self._round[0] != key:
            self._round = (key, set(), Counter(), Counter(), set(), Counter())
        return self._round

    def note_choice(self, request, action, destination=None):
        """Remember an executed dispatch choice for later requests this round."""
        if not self.exclusive or request.role != "dispatcher" or action is None:
            return
        _, decided, empty, loaded, ports, booked = self.dispatch_round(request)
        decided.add(request.owner)
        vehicle = request.observation["agvs"][request.owner]
        (empty if vehicle["job"] is None else loaded)[action.owner] += 1
        ports.add((vehicle["job"] is not None, action.port))
        if self.joint and vehicle["job"] is None:
            if destination is None:
                self.commitments.pop(request.owner, None)
            else:
                booked[self.inputs[destination]] += 1
                self.commitments[request.owner] = (action.owner, destination)

    def expansion(self, request):
        """(candidate index, destination machine) per choice of the last encoding."""
        identity, entries = self._expansion
        return entries if identity == request.identity else None

    def destinations(self, owner, view, claimed=0):
        """Machines able to run the next operation of the job this vehicle gets.

        A source hands out its ready jobs in order, so the vehicle receives the
        one after those already claimed by other empty vehicles heading there.
        """
        ready = view.get("sources", {}).get(owner, {}).get("ready", [])
        if claimed >= len(ready) or ready[claimed] not in view["jobs"]:
            return []
        row = view["jobs"][ready[claimed]]
        op = row.get("next_operation")
        return [
            (machine, row.get("due_at"))
            for machine in self.machine_ids
            if op in self.machine_operations[machine] and machine in self.inputs
        ]

    def joint_entries(self, request, mask):
        """Expand empty-vehicle sources into (source, machine) pairs and their mask."""
        view = request.observation
        _, decided, empty, _, _, booked = self.dispatch_round(request)
        vehicles = view["agvs"]

        def claimed(owner):
            # Same count as the exclusive supply mask: this round and earlier.
            return empty[owner] + sum(
                key != request.owner
                and key not in decided
                and state["job"] is None
                and bool(state["target"])
                and state["target"]["owner"] == owner
                for key, state in vehicles.items()
            )

        def free(machine):
            buffer = self.inputs[machine]
            waiting = sum(row["location"] == buffer for row in view["jobs"].values())
            heading = sum(
                key != request.owner
                and key not in decided
                and state["job"] is not None
                and bool(state["target"])
                and state["target"]["owner"] == buffer
                for key, state in vehicles.items()
            )
            return self.capacities[buffer] - waiting - heading - booked[buffer] > 0

        entries, legal = [], []
        for index, (allowed, candidate) in enumerate(zip(mask, request.candidates)):
            pairs = (
                self.destinations(
                    candidate.action.owner, view, claimed(candidate.action.owner)
                )
                if candidate.action is not None
                else []
            )
            if not pairs:
                entries.append((index, None, None))
                legal.append(allowed)
                continue
            for machine, due in pairs:
                entries.append((index, machine, due))
                legal.append(allowed and free(machine))
        if not any(legal):  # every machine slot booked: keep the source choice
            legal = [mask[index] for index, _, _ in entries]
        return entries, legal

    def joint_features(self, candidate, machine, due, view):
        port = self.port_cells.get(candidate.action.port) if candidate.action else None
        cells = self.buffer_cells.get(self.inputs.get(machine), [])
        distance = (
            min(abs(port[0] - x) + abs(port[1] - y) for x, y in cells)
            if port and cells
            else 0
        )
        scale = self.projection["time_scale"]
        return [
            1.0,
            distance / scale,
            (due - view["tick"]) / scale if due is not None else 0.0,
        ]

    def exclusive_mask(self, request, mask):
        """Supply and destination slots taken earlier this round stay taken."""
        view = request.observation
        vehicles = view["agvs"]
        _, decided, empty, loaded, _, _ = self.dispatch_round(request)
        loaded_now = vehicles[request.owner]["job"] is not None
        inbound = self.inbound_ports(request) if self.idle_fallback else set()

        def heading(owner, carrying):
            # Earlier-tick targets of vehicles not yet re-decided this round.
            return sum(
                key != request.owner
                and key not in decided
                and (state["job"] is not None) == carrying
                and bool(state["target"])
                and state["target"]["owner"] == owner
                for key, state in vehicles.items()
            )

        def open_(candidate):
            if candidate.action is None:
                return False
            owner = candidate.action.owner
            if not loaded_now:
                if self.idle_fallback and candidate.action.port in inbound:
                    return False
                source = view.get("sources", {}).get(owner, {})
                available = max(len(source.get("ready", [])), source.get("supply", 0))
                return available > heading(owner, False) + empty[owner]
            features = candidate.features
            if not features[9]:  # unbounded destination
                return True
            free = round(features[8] * 100) - round(features[7] * 100)
            return free > heading(owner, True) + loaded[owner]

        return [m and open_(c) for m, c in zip(mask, request.candidates)]

    def inbound_ports(self, request):
        """Ports another vehicle in the same load state heads for, this round included."""
        _, decided, _, _, ports, _ = self.dispatch_round(request)
        vehicles = request.observation["agvs"]
        carrying = vehicles[request.owner]["job"] is not None
        return {port for loaded, port in ports if loaded == carrying} | {
            state["target"]["port"]
            for key, state in vehicles.items()
            if key != request.owner
            and key not in decided
            and (state["job"] is not None) == carrying
            and state["target"]
        }

    def park_mask(self, request, mask):
        """Nothing to fetch: only the nearest port no other empty vehicle targets."""
        inbound = self.inbound_ports(request)
        free = [
            (c.features[1], i)
            for i, (m, c) in enumerate(zip(mask, request.candidates))
            if m and c.action is not None and c.action.port not in inbound
        ]
        if not free:
            return [False] * len(mask)
        nearest = min(free)[1]
        return [i == nearest for i in range(len(mask))]

    def release_mask(self, request, mask):
        """A source releases a job for the machine a waiting vehicle chose."""
        view = request.observation
        wanted = Counter()
        for key, (source, machine) in self.commitments.items():
            state = view["agvs"].get(key)
            target = state and state["target"]
            if (
                source == request.owner
                and state["job"] is None
                and not state.get("service")
                and target
                and target["owner"] == source
                and tuple(state.get("cell") or ())
                == self.port_cells.get(target["port"])
            ):
                wanted.update(self.machine_operations[machine])
        for job in request.prefix:
            wanted[view["jobs"].get(job, {}).get("next_operation")] -= 1
        keep = [
            m and wanted[view["jobs"].get(c.identity, {}).get("next_operation")] > 0
            for m, c in zip(mask, request.candidates)
        ]
        return keep if any(keep) else mask

    def dispatch_mask(self, request, mask):
        """Narrow the legal dispatcher set; never empty it."""
        view = request.observation
        vehicles = view["agvs"]
        if not request.candidates or request.owner not in vehicles:
            return mask  # value-only requests carry no vehicle choice
        vehicle = vehicles[request.owner]
        if self.exclusive:
            exclusive = self.exclusive_mask(request, mask)
            if not any(exclusive) and self.idle_fallback and vehicle["job"] is None:
                exclusive = self.park_mask(request, mask)
            mask = exclusive if any(exclusive) else mask
        if (
            self.joint
            and vehicle["job"] is not None
            and request.owner in self.commitments
        ):
            # First loaded request after a joint choice: deliver to that machine.
            _, machine = self.commitments.pop(request.owner)
            keep = [
                m
                and c.action is not None
                and c.action.owner == self.inputs[machine]
                and not c.features[10]
                for m, c in zip(mask, request.candidates)
            ]
            if any(keep):
                return keep
        keep = mask
        # The exclusive mask already counts supply, including this round.
        if vehicle["job"] is None and self.supply_mask and not self.exclusive:
            sources = view.get("sources", {})

            def open_supply(candidate):
                if candidate.action is None:
                    return False
                owner = candidate.action.owner
                source = sources.get(owner, {})
                claimed = sum(
                    key != request.owner
                    and state["job"] is None
                    and bool(state["target"])
                    and state["target"]["owner"] == owner
                    for key, state in vehicles.items()
                )
                available = max(len(source.get("ready", [])), source.get("supply", 0))
                return available > claimed

            keep = [m and open_supply(c) for m, c in zip(mask, request.candidates)]
        elif vehicle["job"] is not None and self.commit_mask and vehicle["target"]:
            current = vehicle["target"]
            keep = [
                m
                and c.action is not None
                and c.action.owner == current["owner"]
                and c.action.port == current["port"]
                and not c.features[10]
                for m, c in zip(mask, request.candidates)
            ]
        return keep if any(keep) else mask

    def encode(self, request):
        view, factory = request.observation, self.factory
        scale, count = self.projection["time_scale"], self.projection["count_scale"]
        core_scale = view.get("feature_time_scale", 100)
        features = []
        for candidate in request.candidates:
            values = list(candidate.features)
            timed = (
                (1, 3, 8)
                if request.role in ("machine", "buffer")
                else (1,)
                if request.role == "dispatcher"
                else (2,)
            )
            for position in timed:
                values[position] *= core_scale / scale
            job = (
                candidate.action[0]
                if request.role == "machine"
                else candidate.action
                if request.role == "buffer"
                else None
            )
            row = view["jobs"].get(job, {})
            risk = float(row.get("risk", -1))
            values.extend(
                [
                    risk if risk >= 0 else 0.0,
                    float(risk >= 0),
                    len(row.get("remaining_steps", [])) / count,
                    float(bool(row)),
                ]
            )
            if request.role == "dispatcher":
                target = candidate.action
                cargo = view["agvs"][request.owner]["job"]
                values.extend(self.semantic_features(cargo, view, target=target))
                if self.machine_features:
                    values.extend(
                        self.target_machine(target.owner if target else None, view)
                    )
            else:
                values.extend(
                    self.semantic_features(
                        job, view, after_start=request.role == "machine"
                    )
                )
                if self.machine_features:
                    values.extend([0.0] * MACHINE_FEATURES)
            features.append(values)
        prefix = np.asarray(
            [self.job_prefix(job, view) for job in request.prefix], dtype=np.float32
        ).reshape(-1, self.candidate_width)
        roles = ("machine", "buffer", "dispatcher", "mover")
        vehicles = view["agvs"]
        state = vehicles.get(request.owner, {})
        target = state.get("target") or {}
        reservation = state.get("reservation")
        location = state.get("cell", (0, 0))
        header = [
            view["tick"] / scale,
            len(view["completed"]) / count,
            len(view["released"]) / count,
            len(view["jobs"]) / count,
            *[float(request.role == role) for role in roles],
            request.count / count,
            len(request.prefix) / count,
            float(state.get("job") is not None),
            float(bool(target)),
            float(reservation is not None),
            float(state.get("service") is not None),
            (view["tick"] - state["reservation_tick"]) / scale
            if state.get("reservation_tick") is not None
            else 0.0,
            float(bool(state.get("feedback"))),
        ]
        context = list(header)
        context.extend(float(owner == request.owner) for owner in self.owners)
        context.extend(float(owner == target.get("owner")) for owner in self.owners)
        context.extend(float(owner == reservation) for owner in self.owners)
        context.extend(float(port == target.get("port")) for port in self.port_ids)
        occupied = {tuple(v["cell"]) for v in vehicles.values()}
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = location[0] + dx, location[1] + dy
                outside = not (
                    0 <= cell[0] < factory.grid.width
                    and 0 <= cell[1] < factory.grid.height
                )
                context.extend(
                    (
                        float(outside or cell in self.solids),
                        float(cell in occupied and cell != tuple(location)),
                    )
                )
        for owner in self.agv_ids:
            vehicle = vehicles[owner]
            context.extend(
                (
                    vehicle["cell"][0] / factory.grid.width,
                    vehicle["cell"][1] / factory.grid.height,
                    float(vehicle["job"] is not None),
                    float(bool(vehicle.get("target"))),
                    float(vehicle.get("service") is not None),
                )
            )
        for owner in self.machine_ids:
            machine = view["machines"][owner]
            context.extend(
                (
                    float(machine["down"]),
                    *[
                        float(machine["status"] == value)
                        for value in ("IDLE", "READY", "PROCESSING", "BLOCKED")
                    ],
                    machine["nominal"] / scale,
                    machine["elapsed"] / scale,
                    float(machine["job"] is not None),
                )
            )
        for owner in self.owners:
            rows = [row for row in view["jobs"].values() if row["location"] == owner]
            supply = view.get("sources", {}).get(owner, {})
            capacity = self.capacities.get(owner)
            context.extend(
                (
                    len(rows) / count,
                    len(supply.get("ready", [])) / count,
                    supply.get("supply", 0) / count,
                    supply.get("reserved", 0) / count,
                    capacity / count if capacity is not None else 0.0,
                    float(capacity is None),
                    sum(row["quality"] == "PASS" for row in rows) / count,
                    sum(row["quality"] == "UNKNOWN" for row in rows) / count,
                    sum(row["quality"] == "FAIL" for row in rows) / count,
                    *[
                        sum(row["next_operation"] == kind for row in rows) / count
                        for kind in self.operation_types
                    ],
                )
            )
        context.extend(self.map)
        if len(context) != self.base_context_size:
            raise ValueError("public encoder dimension mismatch")
        from smartsom.learning.extensions import PublicObservation

        public = PublicObservation(
            request,
            tuple(context),
            (("state", tuple(context)),),
            self.extension_role,
            request.owner,
            self.training,
        )
        before = self.state_dict() if not self.training else None
        encoded = self.extensions.encode(public)
        if before is not None and self.state_dict() != before:
            self.load_state_dict(before)
            raise ValueError(
                "frozen observation extension changed normalization statistics"
            )
        space = self.extensions.spaces[self.extension_role]
        encoded = (
            np.asarray(encoded).reshape(-1)
            if space.vector
            else np.concatenate(
                [np.asarray(encoded[k]).reshape(-1) for k, _ in space.fields]
            )
        )
        mask = [c.legal for c in request.candidates]
        if request.role == "dispatcher" and (
            self.supply_mask or self.commit_mask or self.exclusive
        ):
            mask = self.dispatch_mask(request, mask)
        if self.joint and request.role == "buffer":
            mask = self.release_mask(request, mask)
        entries = None
        if (
            self.joint
            and request.role == "dispatcher"
            and request.candidates
            and request.owner in vehicles
            and vehicles[request.owner]["job"] is None
        ):
            entries, mask = self.joint_entries(request, mask)
            features = [
                features[index][:-MACHINE_FEATURES]
                + self.target_machine(self.inputs[machine], view)
                + self.joint_features(request.candidates[index], machine, due, view)
                if machine
                else features[index] + [0.0] * JOINT_FEATURES
                for index, machine, due in entries
            ]
        elif self.joint:
            features = [row + [0.0] * JOINT_FEATURES for row in features]
        self._expansion = (
            (request.identity, [(index, machine) for index, machine, _ in entries])
            if entries is not None
            else (None, None)
        )
        candidates = np.asarray(features, dtype=np.float32).reshape(
            -1, self.candidate_width
        )
        result = {
            "context": np.asarray(encoded, dtype=np.float32),
            "candidates": candidates,
            "prefix": prefix,
            "mask": np.asarray(mask, dtype=bool),
            "feature_width": np.asarray(self.candidate_width),
        }
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("nonfinite public observation")
        return result

    def empty(self, role, owner, view):
        from smartsom.domain.production_decisions import DecisionRequest

        request = DecisionRequest(view["tick"], "value", role, owner, (), view)
        result = self.encode(request)
        return result


class CandidateNetwork(nn.Module):
    DIAGNOSTIC_CONTRACT = {
        "schema": "smartsom.candidate-network/v1",
        "scoring": "masked per-candidate score",
        "critic_candidate_pool": "zeroed",
        "critic_prefix": "zeroed",
    }

    def __init__(
        self,
        context_size,
        network,
        provider,
        algorithm="ppo",
        *,
        central=False,
        central_private_end=16,
        candidate_width=CANDIDATE_WIDTH,
    ):
        super().__init__()
        self.context_size, self.algorithm, self.central = (
            context_size,
            algorithm,
            central,
        )
        self.central_private_end = central_private_end
        self.candidate_width = candidate_width
        self.actor_prefix = nn.GRU(candidate_width, PREFIX_WIDTH, batch_first=True)
        actor_space = ObservationSpace(
            vector=(context_size + PREFIX_WIDTH + candidate_width,)
        )
        if algorithm == "dqn":
            selected = QNetwork.model_validate_json(__import__("json").dumps(network))
            self.actor = _Branch(actor_space, 1, selected.q, provider)
            self.critic = None
        else:
            selected = ActorCriticSpec.model_validate_json(
                __import__("json").dumps(network)
            )
            self.actor = _Branch(actor_space, 1, selected.actor, provider)
            self.critic_prefix = nn.GRU(candidate_width, PREFIX_WIDTH, batch_first=True)
            critic_space = ObservationSpace(
                vector=(context_size + PREFIX_WIDTH + 2 * candidate_width,)
            )
            self.critic = _Branch(critic_space, 1, selected.critic, provider)
        self.action_mode = "softmax" if algorithm == "dqn" else selected.action_mode
        self.role_action_modes = (
            {} if algorithm == "dqn" else dict(selected.role_action_modes)
        )

    def prefix_embedding(self, prefix, lengths, branch):
        if prefix.shape[1] == 0:
            return torch.zeros((len(prefix), PREFIX_WIDTH), device=prefix.device)
        outputs, _ = branch(prefix)
        indexes = (lengths - 1).clamp(min=0)
        last = outputs[torch.arange(len(prefix), device=prefix.device), indexes]
        return torch.where((lengths > 0).unsqueeze(-1), last, torch.zeros_like(last))

    def forward(self, observations):
        context = observations["context"].float()
        candidates = observations["candidates"].float()
        prefix = observations["prefix"].float()
        mask = observations["mask"].bool()
        lengths = observations["prefix_length"].long()
        actor_prefix = self.prefix_embedding(prefix, lengths, self.actor_prefix)
        expanded = (
            torch.cat((context, actor_prefix), -1)
            .unsqueeze(1)
            .expand(-1, candidates.shape[1], -1)
        )
        scores = self.actor(torch.cat((expanded, candidates), -1)).squeeze(-1)
        if self.role_action_modes:
            # Context positions 4..6 are the public machine/buffer/dispatcher
            # role flags of the request header.
            transformed = {}
            mixed = scores
            for index, role in enumerate(("machine", "buffer", "dispatcher")):
                mode = self.role_action_modes.get(role, self.action_mode)
                if mode == "softmax":
                    continue
                if mode not in transformed:
                    transformed[mode] = sequential_binary_log_probabilities(
                        scores, mask, uniform=mode == "sequential_binary_uniform"
                    )
                rows = (context[:, 4 + index] > 0.5).unsqueeze(-1)
                mixed = torch.where(rows, transformed[mode], mixed)
            scores = mixed
        elif self.action_mode != "softmax":
            scores = sequential_binary_log_probabilities(
                scores, mask, uniform=self.action_mode == "sequential_binary_uniform"
            )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if self.critic is None:
            return scores, scores.max(-1).values
        weights = mask.float().unsqueeze(-1)
        mean = (candidates * weights).sum(1) / weights.sum(1).clamp(min=1)
        maximum = candidates.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        critic_context = context.clone()
        critic_context[:, 8:10] = 0
        mean, maximum = torch.zeros_like(mean), torch.zeros_like(maximum)
        if self.central:
            critic_context = context.clone()
            critic_context[:, 4 : self.central_private_end] = 0
            # Central value is a whole-boundary value, not a chosen role's head.
            mean, maximum = torch.zeros_like(mean), torch.zeros_like(maximum)
        critic_prefix = self.prefix_embedding(prefix, lengths, self.critic_prefix)
        critic_prefix = torch.zeros_like(critic_prefix)
        values = self.critic(
            torch.cat((critic_context, critic_prefix, mean, maximum), -1)
        ).squeeze(-1)
        return scores, values


def sequential_binary_log_probabilities(scores, mask, *, uniform=False):
    """Log-probabilities of "first yes" over legal candidates in their order.

    Each legal candidate i says yes with sigmoid(score_i); illegal ones are
    skipped; the last legal candidate always says yes, so the result is a
    normalized distribution over legal candidates. With uniform=True a fixed
    prior gives the k-th legal candidate yes-odds 1/(r-1), where r legal
    candidates remain, so zero scores select every legal candidate with 1/m.
    """
    positions = torch.arange(scores.shape[-1], device=scores.device)
    last = torch.where(mask, positions, -1).max(-1, keepdim=True).values
    final = positions == last
    if uniform:
        remaining = mask.sum(-1, keepdim=True) - torch.cumsum(mask, -1) + 1
        prior = -torch.log((remaining - 1).clamp(min=1).to(scores.dtype))
        scores = scores + torch.where(mask & ~final, prior, torch.zeros_like(scores))
    yes = torch.where(final, torch.zeros_like(scores), F.logsigmoid(scores))
    no = torch.where(mask & ~final, F.logsigmoid(-scores), torch.zeros_like(scores))
    return yes + torch.cumsum(no, -1) - no


def default_network(algorithm):
    return primitive(QNetwork() if algorithm == "dqn" else ActorCriticSpec())


def pad_inputs(inputs):
    """Pad to the batch maximum only; keep every candidate and conditional prefix."""
    size = len(inputs)
    width = int(inputs[0].get("feature_width", CANDIDATE_WIDTH))
    if any(int(v.get("feature_width", CANDIDATE_WIDTH)) != width for v in inputs):
        raise ValueError("mixed candidate encoding dimensions in one batch")
    candidate_count = max(1, max(len(v["candidates"]) for v in inputs))
    prefix_count = max(0, max(len(v["prefix"]) for v in inputs))
    context = np.stack([v["context"] for v in inputs])
    candidates = np.zeros((size, candidate_count, width), np.float32)
    prefix = np.zeros((size, prefix_count, width), np.float32)
    mask = np.zeros((size, candidate_count), bool)
    lengths = np.zeros(size, np.int64)
    for i, value in enumerate(inputs):
        n, p = len(value["candidates"]), len(value["prefix"])
        if n:
            candidates[i, :n] = value["candidates"]
        if p:
            prefix[i, :p] = value["prefix"]
        mask[i, :n] = value["mask"]
        if n == 0:
            mask[i, 0] = True  # value-only tensor padding, never an action request
        lengths[i] = p
    return {
        "context": context,
        "candidates": candidates,
        "prefix": prefix,
        "mask": mask,
        "prefix_length": lengths,
    }


def tensor_inputs(inputs, device="cpu"):
    return {k: torch.as_tensor(v, device=device) for k, v in pad_inputs(inputs).items()}


def packet_arrays(rows):
    """One row/ratio per owner tick; conditional choices are a separate dimension."""
    maximum_steps = max(1, max(len(r["inputs"]) for r in rows))
    flat, actions, step_mask = [], [], []
    for row in rows:
        inputs = list(row["inputs"]) or [row["value_input"]]
        selected = list(row["actions"]) or [0]
        active = [True] * len(row["actions"])
        for step in range(maximum_steps):
            flat.append(inputs[step] if step < len(inputs) else row["value_input"])
            actions.append(selected[step] if step < len(selected) else 0)
            step_mask.append(active[step] if step < len(active) else False)
    padded = pad_inputs(flat)
    obs = {
        k: value.reshape(len(rows), maximum_steps, *value.shape[1:])
        for k, value in padded.items()
    }
    batch = {
        "obs": obs,
        "value_obs": pad_inputs([r["value_input"] for r in rows]),
        "actions": np.asarray(actions, np.int64).reshape(len(rows), maximum_steps),
        "step_mask": np.asarray(step_mask, bool).reshape(len(rows), maximum_steps),
        "old_log_probability": np.asarray(
            [r["log_probability"] for r in rows], np.float32
        ),
        "actor_mask": np.asarray([r["actor_mask"] for r in rows], bool),
        "value_mask": np.asarray([r.get("value_mask", True) for r in rows], bool),
        "advantages": np.asarray([r["advantage"] for r in rows], np.float32),
        "raw_advantages": np.asarray(
            [r.get("raw_advantage", float("nan")) for r in rows], np.float32
        ),
        "returns": np.asarray([r["return"] for r in rows], np.float32),
    }
    return batch
