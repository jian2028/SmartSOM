"""Opt-in physical-job observations for native PPO and DQN.

Installation is process-local and must precede policy construction. The explicit
schema identity, rather than this file's bytes, defines package compatibility.
"""

import copy
import math
import sys
from dataclasses import replace

import numpy as np
import torch

from smartsom.learning.production_models import (
    CandidateNetwork as NativeCandidateNetwork,
)
from smartsom.learning.production_models import PublicEncoder as NativePublicEncoder

SCHEMA = "smartsom.physical-job-observation/v1"
QUALIFIED_SOURCE_SHA256 = (
    "5c4da1dd5da2276d95010aa325493c1ff3d740234cf0da401365bce13b4b9b02"
)
IDENTITY = SCHEMA
INCLUDE_INSPECTION = None  # explicit common contract required before installation
_INSTALLED = False
CRITIC_FIELDS = (
    "urgency_min",
    "urgency_mean",
    "urgency_negative_fraction",
    "remaining_mean",
    "remaining_max",
    "work_pressure",
    "remaining_fraction_mean",
    "has_visible_jobs",
)


def critic_features(view, factory, capacity, time_scale):
    """Current physical unfinished jobs only; no horizon or future-demand count."""
    rows = [processing_features(job, view, factory, time_scale) for job in view["jobs"]]
    if not rows:
        return np.zeros(len(CRITIC_FIELDS), dtype=np.float32)
    urgency, fraction, remaining = np.asarray(rows, dtype=np.float64).T
    return np.asarray(
        (
            urgency.min(),
            urgency.mean(),
            (urgency < 0).mean(),
            remaining.mean(),
            remaining.max(),
            remaining.sum() / capacity,
            fraction.mean(),
            1.0,
        ),
        dtype=np.float32,
    )


def _normal_reference(step, factory):
    kind = step["operation_type"]
    overrides = dict(step.get("machine_nominal_ticks", {}))
    times = []
    for machine in factory.machines:
        if kind in machine.operation_types:
            normal = next(
                q for q in machine.quality_modes if q.quality_mode_id == "normal"
            )
            times.append(
                float(overrides.get(machine.machine_id, step["nominal_ticks"]))
                * float(normal.time_scale)
                / float(machine.processing_rate_multiplier)
            )
    if not times or not all(math.isfinite(x) and x > 0 for x in times):
        raise ValueError("invalid released-route processing reference")
    return sum(times) / len(times)


def processing_features(job, view, factory, time_scale=100.0):
    row = view["jobs"][job]
    if row["demand"] not in view["released"]:
        raise ValueError("unreleased job cannot enter physical-job observation encoder")
    original = float(row["physical_job_original_reference_work"])
    if not math.isfinite(original) or original <= 0:
        raise ValueError("original reference work must be positive")
    steps = row["remaining_steps"]
    reference = [_normal_reference(step, factory) for step in steps]
    if reference:
        machine = view["machines"].get(row["location"])
        if machine and machine["job"] == job and machine["status"] == "PROCESSING":
            nominal = float(machine["nominal"])
            if nominal <= 0:
                raise ValueError("current public nominal duration must be positive")
            reference[0] *= max(
                0.0, min(1.0, 1.0 - float(machine["elapsed"]) / nominal)
            )
    remaining = sum(reference)
    if not math.isfinite(remaining) or remaining < 0 or remaining > original + 1e-9:
        raise ValueError("inconsistent original/remaining reference work")
    return (
        (float(row["due_at"]) - float(view["tick"]) - remaining) / time_scale,
        remaining / original,
        remaining / time_scale,
    )


def physical_capacities(factory):
    capacities = {m.machine_id: 1 for m in factory.machines}
    capacities.update({a.agv_id: 1 for a in factory.agvs})
    for buffer in factory.buffers:
        if buffer.role not in ("system_input", "machine_pre", "machine_post"):
            continue
        capacity = (
            sum(slot.capacity for slot in buffer.storage.slots)
            if hasattr(buffer.storage, "slots")
            else buffer.storage.capacity
        )
        capacities[buffer.buffer_id] = capacity
    for station in factory.inspection_stations:
        capacities[station.inspection_station_id] = sum(
            slot.capacity for slot in station.slots
        )
    return capacities


class PhysicalJobEncoder(NativePublicEncoder):
    def __init__(self, factory, projection, **kwargs):
        if INCLUDE_INSPECTION is None:
            raise ValueError(
                "physical-job observation inspection-capacity contract is not frozen"
            )
        super().__init__(factory, projection, **kwargs)
        if kwargs.get("observation"):
            raise ValueError(
                "physical-job observation override cannot silently compose another observation transform"
            )
        self.physical = physical_capacities(factory)
        inputs = [b for b in factory.buffers if b.role == "system_input"]
        if len(inputs) != 1 or self.physical[inputs[0].buffer_id] != {
            8: 10,
            16: 20,
        }.get(len(factory.machines)):
            raise ValueError(
                "physical-job observation requires its explicitly configured finite input"
            )
        stations = {s.inspection_station_id for s in factory.inspection_stations}
        self.global_owners = set(self.physical) - (
            set() if INCLUDE_INSPECTION else stations
        )
        if any(self.physical[owner] is None for owner in self.global_owners):
            raise ValueError(
                "global occupancy ratio unavailable with an unlimited included owner"
            )
        self.global_capacity = sum(self.physical[owner] for owner in self.global_owners)
        start = (
            16
            + len(self.owners) * 3
            + len(self.port_ids)
            + 50
            + len(self.agv_ids) * 5
            + len(self.machine_ids) * 8
        )
        block = 9 + len(self.operation_types)
        self.owner_start, self.owner_width = start, block
        keep = list(range(3, start))
        for index, owner in enumerate(self.owners):
            offset = start + index * block
            # End sinks retain physical capacity/availability flags, never historical job counts.
            local = (
                range(block)
                if owner in self.physical and self.physical[owner] is not None
                else (4, 5)
            )
            keep.extend(offset + field for field in local)
        keep.extend(range(start + len(self.owners) * block, self.base_context_size))
        self.kept_context = np.asarray(keep, dtype=np.int64)
        self.context_size = len(keep)
        self.central_private_end -= 3
        self.context_size += len(CRITIC_FIELDS)
        # A common width also supports central/mixed-role controllers. The eight
        # appended cargo summary fields are zero for other roles and empty AGVs.
        self.candidate_width += 8

    def job_prefix(self, job, observation):
        values = super().job_prefix(job, observation)
        urgency, fraction, magnitude = processing_features(
            job, observation, self.factory, self.projection["time_scale"]
        )
        values[0], values[1], values[3] = fraction, urgency, magnitude
        return values

    def encode(self, request):
        view = dict(request.observation)
        view["jobs"] = {
            key: row
            for key, row in view["jobs"].items()
            if row["location"] in self.physical
        }
        native = copy.copy(self)
        native.candidate_width -= 8
        encoded = NativePublicEncoder.encode(native, replace(request, observation=view))
        context = encoded["context"].copy()
        context[3] = (
            sum(row["location"] in self.global_owners for row in view["jobs"].values())
            / self.global_capacity
        )
        for index, owner in enumerate(self.owners):
            capacity = self.physical.get(owner)
            if capacity is None:
                continue
            offset = self.owner_start + index * self.owner_width
            rows = [row for row in view["jobs"].values() if row["location"] == owner]
            supply = view.get("sources", {}).get(owner, {})
            counts = (
                len(rows),
                len(supply.get("ready", [])),
                supply.get("supply", 0),
                supply.get("reserved", 0),
            )
            context[offset : offset + 4] = np.asarray(counts) / capacity
            context[offset + 6 : offset + 9] = [
                sum(row["quality"] == quality for row in rows) / capacity
                for quality in ("PASS", "UNKNOWN", "FAIL")
            ]
            context[offset + 9 : offset + self.owner_width] = [
                sum(row["next_operation"] == kind for row in rows) / capacity
                for kind in self.operation_types
            ]
        encoded["context"] = np.concatenate(
            (
                context[self.kept_context],
                critic_features(
                    view,
                    self.factory,
                    self.global_capacity,
                    self.projection["time_scale"],
                ),
            )
        )
        if request.role in ("machine", "buffer"):
            for index, candidate in enumerate(request.candidates):
                job = (
                    candidate.action[0]
                    if request.role == "machine"
                    else candidate.action
                )
                urgency, fraction, magnitude = processing_features(
                    job, view, self.factory, self.projection["time_scale"]
                )
                encoded["candidates"][index, (0, 1, 3)] = (fraction, urgency, magnitude)
        cargo_summary = np.zeros((len(request.candidates), 8), dtype=np.float32)
        if request.role == "dispatcher":
            cargo = view["agvs"][request.owner]["job"]
            if cargo is not None:
                # The identical approved Machine/Buffer job summary, including
                # public risk/availability; never latent sampled defect truth.
                summary = self.job_prefix(cargo, view)
                cargo_summary[:] = summary[:8]
                encoded["candidates"][:, 12:16] = summary[12:16]
        encoded["candidates"] = np.concatenate(
            (encoded["candidates"], cargo_summary), axis=1
        )
        encoded["prefix"] = np.concatenate(
            (
                encoded["prefix"],
                np.zeros((len(encoded["prefix"]), 8), dtype=np.float32),
            ),
            axis=1,
        )
        encoded["feature_width"] = np.asarray(self.candidate_width)
        return encoded


class PhysicalJobNetwork(NativeCandidateNetwork):
    def forward(self, observations):
        context = observations["context"].float()
        candidates = observations["candidates"].float()
        prefix = observations["prefix"].float()
        mask = observations["mask"].bool()
        lengths = observations["prefix_length"].long()
        actor_prefix = self.prefix_embedding(prefix, lengths, self.actor_prefix)
        actor_context = context.clone()
        actor_context[:, -len(CRITIC_FIELDS) :] = 0
        expanded = (
            torch.cat((actor_context, actor_prefix), -1)
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
        critic_context[:, 5:7] = (
            0  # native 8:10 shifted by exactly three header deletions
        )
        mean, maximum = torch.zeros_like(mean), torch.zeros_like(maximum)
        if self.central:
            critic_context = context.clone()
            critic_context[:, 1 : self.central_private_end] = 0
            mean, maximum = torch.zeros_like(mean), torch.zeros_like(maximum)
        critic_prefix = torch.zeros_like(
            self.prefix_embedding(prefix, lengths, self.critic_prefix)
        )
        values = self.critic(
            torch.cat((critic_context, critic_prefix, mean, maximum), -1)
        ).squeeze(-1)
        return scores, values


def validate_package_identity(metadata):
    """Reject another observation schema before loading model weights."""
    if not _INSTALLED or metadata.get("physical_job_encoder") != IDENTITY:
        raise ValueError(
            "old/different physical-job observation encoder package rejected"
        )


def install(*, include_inspection):
    """Local process-only installation, called before preparation/initial models."""
    global INCLUDE_INSPECTION, IDENTITY, _INSTALLED
    if type(include_inspection) is not bool:
        raise ValueError("explicit inspection inclusion decision required")
    if _INSTALLED:
        if INCLUDE_INSPECTION != include_inspection:
            raise ValueError("cannot change encoder contract in a live process")
        return IDENTITY
    INCLUDE_INSPECTION = include_inspection
    IDENTITY = SCHEMA + "/inspection=" + str(include_inspection)
    from smartsom.engine.production_protocol import ProductionProtocol
    from smartsom.learning import production_inference, production_models

    old_view = ProductionProtocol.public_view

    def public_view(protocol):
        view = old_view(protocol)
        physical = physical_capacities(protocol.core.factory)
        for job, row in view["jobs"].items():
            if row["location"] not in physical:
                continue
            if row["demand"] not in protocol.core.released:
                raise ValueError("unreleased original route cannot be consulted")
            demand = protocol.core.demands[row["demand"]]
            steps = [
                dict(
                    operation_type=s.operation_type,
                    nominal_ticks=s.nominal_ticks,
                    machine_nominal_ticks=dict(s.machine_nominal_ticks),
                )
                for s in demand.steps
            ]
            row["physical_job_original_reference_work"] = sum(
                _normal_reference(step, protocol.core.factory) for step in steps
            )
        return view

    ProductionProtocol.public_view = public_view
    production_models.PublicEncoder = production_inference.PublicEncoder = (
        PhysicalJobEncoder
    )
    production_models.CandidateNetwork = production_inference.CandidateNetwork = (
        PhysicalJobNetwork
    )
    for name in (
        "smartsom.learning.production_rllib_v3",
        "smartsom.learning.production_sb3_v3",
    ):
        if name in sys.modules:
            sys.modules[name].CandidateNetwork = PhysicalJobNetwork
    original_read = production_inference.read_package

    def read_package(path):
        return original_read(path, metadata_validator=validate_package_identity)

    production_inference.read_package = read_package
    from smartsom.experiments import composable

    original_policies = composable.policies_for

    def policies_for(*args, **kwargs):
        policies, learners = original_policies(*args, **kwargs)
        for policy in policies.values():
            if hasattr(policy, "metadata"):
                policy.metadata["physical_job_encoder"] = IDENTITY
        return policies, learners

    composable.policies_for = policies_for
    _INSTALLED = True
    return IDENTITY
