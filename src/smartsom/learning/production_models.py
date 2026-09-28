"""Optional ragged candidate networks; masks never truncate a candidate pool."""

import numpy as np
import torch
from torch import nn

from smartsom.config.codec import primitive
from smartsom.config.extensions import ActorCriticSpec
from smartsom.config.policies import QNetwork
from smartsom.domain.factory_design import occupied_cells
from smartsom.learning.extensions import ObservationSpace
from smartsom.learning.torch_extensions import _Branch

CANDIDATE_WIDTH = 16
PREFIX_WIDTH = 32


class PublicEncoder:
    def __init__(
        self,
        factory,
        projection,
        *,
        observation=None,
        provider="rllib.resource_ppo",
        role="machine",
        training=True,
    ):
        self.factory, self.projection = factory, dict(projection)
        self.role, self.training = role, training
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
        )
        self.machine_ids = sorted(m.machine_id for m in factory.machines)
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
        ]

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
            else:
                values.extend(
                    self.semantic_features(
                        job, view, after_start=request.role == "machine"
                    )
                )
            features.append(values)
        candidates = np.asarray(features, dtype=np.float32).reshape(
            -1, self.candidate_width
        )
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
        result = {
            "context": np.asarray(encoded, dtype=np.float32),
            "candidates": candidates,
            "prefix": prefix,
            "mask": np.asarray([c.legal for c in request.candidates], dtype=bool),
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
        "returns": np.asarray([r["return"] for r in rows], np.float32),
    }
    return batch
