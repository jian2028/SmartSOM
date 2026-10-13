"""Registered public field computations over a restricted detached input.

No simulator or arbitrary YAML expression enters a component. DQN stores only
actor/Q context; PPO constructs compact, independently sized actor and critic inputs.
"""

import copy
import json
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import torch
from torch import nn

from smartsom.config.codec import digest, primitive
from smartsom.config.extensions import ActorCriticSpec
from smartsom.config.observation_contracts import SCHEMA
from smartsom.config.policies import QNetwork
from smartsom.learning.extensions import ObservationSpace
from smartsom.learning.physical_job_observation import (
    CRITIC_FIELDS,
    physical_capacities,
)
from smartsom.learning.production_models import (
    PREFIX_WIDTH,
    CandidateNetwork,
    PublicEncoder,
)
from smartsom.learning.torch_extensions import _Branch

JOB_FIELDS = (
    "remaining_fraction",
    "urgency",
    "remaining_work",
    "priority",
    "age",
    "quality_presence",
    "quality_pass",
    "risk",
    "risk_presence",
    "remaining_steps",
    "job_presence",
)
JOB_KEYS = (
    "demand",
    "location",
    "quality",
    "since",
    "remaining_steps",
    "original_steps",
    "due_at",
    "priority",
    "risk",
    "next_operation",
)
MACHINE_KEYS = ("job", "down", "status", "nominal", "elapsed", "mode")
AGV_KEYS = ("job", "cell", "service", "feedback")


def restricted_view(view, owner):
    """Keep released task state and real resources; exclude side-channel ledgers."""
    released = set(view["released"])
    jobs = {}
    for key, row in view["jobs"].items():
        if row["demand"] not in released:
            raise ValueError("unreleased job entered registered observation")
        jobs[key] = {name: copy.deepcopy(row[name]) for name in JOB_KEYS if name in row}
        for name in ("remaining_steps", "original_steps"):
            jobs[key][name] = [
                {
                    field: copy.deepcopy(step[field])
                    for field in (
                        "operation_type",
                        "nominal_ticks",
                        "reference_ticks",
                        "machine_nominal_ticks",
                    )
                    if field in step
                }
                for step in row[name]
            ]
    vehicles = {}
    for key, row in view["agvs"].items():
        vehicles[key] = {
            name: copy.deepcopy(row[name]) for name in AGV_KEYS if name in row
        }
        # Only this decision owner receives its target or target-presence bit.
        if key == owner:
            vehicles[key]["target"] = copy.deepcopy(row.get("target"))
        vehicles[key]["service"] = bool(row.get("service"))
        vehicles[key]["feedback"] = bool(row.get("feedback"))
    return {
        "tick": view["tick"],
        "feature_time_scale": view.get("feature_time_scale", 100),
        "released": tuple(sorted(released)),
        "jobs": jobs,
        "agvs": vehicles,
        "machines": {
            key: {
                name: copy.deepcopy(row[name]) for name in MACHINE_KEYS if name in row
            }
            for key, row in view["machines"].items()
        },
        "sources": {
            key: {"ready": list(row.get("ready", ())), "supply": row.get("supply", 0)}
            for key, row in view.get("sources", {}).items()
        },
        "disposing": frozenset(
            job
            for station in view.get("stations", {}).values()
            for job, state in station.get("jobs", {}).items()
            if state.get("status") == "DISPOSING"
        ),
    }


def nominal_ticks(step, machine, mode, rounding):
    """Declared exact work, overrides, rate, mode and physical nominal rounding."""
    work = dict(step.get("machine_nominal_ticks", {})).get(machine.machine_id)
    work = Fraction(
        work
        if work is not None
        else step.get("reference_ticks") or step["nominal_ticks"]
    )
    value = (
        work * Fraction(mode.time_scale) / Fraction(machine.processing_rate_multiplier)
    )
    if rounding == "ceil":
        return max(1, math.ceil(value))
    return max(1, (2 * value.numerator + value.denominator) // (2 * value.denominator))


class PublicFieldsEncoder(PublicEncoder):
    DIAGNOSTIC_CONTRACT = {
        "schema": SCHEMA,
        "actor_scope": "explicit registered context, candidates and conditional prefix",
        "critic_only_fields": list(CRITIC_FIELDS),
        "context_planes": ["compact actor/Q", "compact critic (PPO only)"],
        "quality": "presence+PASS; actionable FAIL rejected",
        "intentions": "own target only; no neighbor has-target or target counts",
        "attempt": "metrics only; absent from every observation component",
        "local_map": "Dispatcher/Mover only, own center, leading applicability bit",
        "unlimited_capacity": "projection count with explicit unlimited flag",
        "nominal": "declared work/overrides/rate/mode/rounding; no realized samples",
    }

    def __init__(self, factory, projection, *, settings, algorithm="ppo", **kwargs):
        super().__init__(factory, projection, **kwargs)
        self.settings = settings
        if algorithm not in ("ppo", "dqn"):
            raise ValueError("unsupported compact observation algorithm")
        self.algorithm = algorithm
        self.observation_identity = (
            SCHEMA + "/" + digest(primitive(settings)) + "/" + algorithm
        )
        self.physical = physical_capacities(factory)
        self.outputs = {
            b.buffer_id for b in factory.buffers if b.role == "system_output"
        }
        self.machines_by_id = {m.machine_id: m for m in factory.machines}
        stations = {s.inspection_station_id for s in factory.inspection_stations}
        self.global_owners = set(self.physical) - (
            set() if settings.include_inspection else stations
        )
        finite = [self.physical[owner] for owner in self.global_owners]
        self.global_unlimited = any(capacity is None for capacity in finite)
        self.global_capacity = sum(capacity or 0 for capacity in finite)
        self.widths = {
            "decision": 11,
            "self_target": 2 + len(self.owners) + len(self.port_ids),
            "local_map": 51,
            "vehicles": 4,
            "machines": 8,
            "inventory": 9 + len(self.operation_types),
            "topology": len(self.map),
            "work_summary": len(CRITIC_FIELDS),
            "job": len(JOB_FIELDS),
            "semantics": self.candidate_width - 16,
            "machine_mode": 5,
            "dispatch_target": 12,
            "cargo": len(JOB_FIELDS),
            "movement": 12,
        }
        self.layout = []
        offset = 0
        sizes = {}
        for branch in ("actor", "critic"):
            start = offset
            if branch == "actor" or algorithm == "ppo":
                for selection in self.settings.context:
                    if selection.visibility not in (branch, "both"):
                        continue
                    width = self._width(selection)
                    self.layout.append(
                        dict(
                            component=primitive(selection),
                            branch=branch,
                            offset=offset,
                            width=width,
                        )
                    )
                    offset += width
            sizes[branch] = offset - start
        self.context_layout = sizes
        if algorithm == "ppo" and sizes["critic"] == 0:
            raise ValueError("compact PPO requires critic-visible context")
        self.context_size = offset
        self.candidate_width = sum(
            self.widths[selection.name] for selection in settings.candidates
        )
        self.central_private_end = 0  # privacy is determined by named components

    def _width(self, selection):
        rows = {
            "vehicles": len(self.agv_ids),
            "machines": len(self.machine_ids),
            "inventory": len(self.owners),
        }.get(selection.name, 1)
        return self.widths[selection.name] * (
            rows if selection.aggregation == "per_entity" else 1
        )

    def state_dict(self):
        return {
            "public_fields": self.observation_identity,
            "context_size": self.context_size,
            "candidate_width": self.candidate_width,
            "layout": copy.deepcopy(self.layout),
            "algorithm": self.algorithm,
            "context_layout": copy.deepcopy(self.context_layout),
        }

    def validate_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("public field observation identity/shape changed")

    def load_state_dict(self, state):
        self.validate_state_dict(state)

    def set_training(self, training):
        self.training = training

    def _scales(self, selection):
        return (
            (self.projection["time_scale"], self.projection["count_scale"])
            if selection.normalization != "raw"
            else (1.0, 1.0)
        )

    def _ratio(self, count, owner, selection):
        if owner in self.outputs:
            return 0.0
        capacity = self.capacities[owner]
        divisor = (
            capacity
            if selection.normalization == "capacity" and capacity is not None
            else self._scales(selection)[1]
        )
        return count / max(1, divisor)

    def _normal(self, step):
        values = [
            nominal_ticks(
                step,
                machine,
                next(
                    mode
                    for mode in machine.quality_modes
                    if mode.quality_mode_id == "normal"
                ),
                self.settings.rounding,
            )
            for machine in self.factory.machines
            if step["operation_type"] in machine.operation_types
        ]
        if not values:
            raise ValueError("no capable machine for released processing reference")
        return sum(values) / len(values)

    def work(self, job, view, scale):
        row = view["jobs"][job]
        if job in view["disposing"]:
            return (0.0, 0.0, 0.0)
        original = sum(self._normal(step) for step in row["original_steps"])
        remaining = [self._normal(step) for step in row["remaining_steps"]]
        machine = view["machines"].get(row["location"])
        if (
            remaining
            and machine
            and machine["job"] == job
            and machine["status"] == "PROCESSING"
        ):
            design = self.machines_by_id[row["location"]]
            mode = next(
                mode
                for mode in design.quality_modes
                if mode.quality_mode_id == machine["mode"]
            )
            duration = nominal_ticks(
                row["remaining_steps"][0], design, mode, self.settings.rounding
            )
            remaining[0] *= max(0.0, min(1.0, 1 - machine["elapsed"] / duration))
        total = sum(remaining)
        if original <= 0 or total < 0 or total > original + 1e-9:
            raise ValueError("invalid original/remaining public reference work")
        return (
            (row["due_at"] - view["tick"] - total) / scale,
            total / original,
            total / scale,
        )

    def _job(self, job, view, selection):
        if job is None:
            return [0.0] * len(JOB_FIELDS)
        row = view["jobs"][job]
        if row["quality"] not in ("UNKNOWN", "PASS") or job in view["disposing"]:
            raise ValueError(
                "FAIL/DISPOSING cannot enter actionable job or cargo fields"
            )
        time, count = self._scales(selection)
        urgency, fraction, magnitude = self.work(job, view, time)
        risk = float(row.get("risk", -1))
        return [
            fraction,
            urgency,
            magnitude,
            row["priority"] / 10,
            (view["tick"] - row["since"]) / time,
            1.0,
            float(row["quality"] == "PASS"),
            max(0, risk),
            float(risk >= 0),
            len(row["remaining_steps"]) / count,
            1.0,
        ]

    def _semantics(self, job, view, selection, *, target=None, after_start=False):
        row = view["jobs"].get(job, {})
        steps = row.get("remaining_steps", ())
        upcoming = steps[1:] if after_start else steps
        kind = upcoming[0]["operation_type"] if upcoming else None
        location = target.owner if target else row.get("location")
        port = target.port if target else None
        time, count = self._scales(selection)
        return [
            *[float(kind == operation) for operation in self.operation_types],
            *[
                sum(step["operation_type"] == operation for step in upcoming) / count
                for operation in self.operation_types
            ],
            *[float(location == owner) for owner in self.owners],
            *[float(port == key) for key in self.port_ids],
            *[
                float(kind in self.machines_by_id[owner].operation_types)
                for owner in self.machine_ids
            ],
            *[
                self._normal_on(upcoming[0], owner) / time
                if upcoming and kind in self.machines_by_id[owner].operation_types
                else 0.0
                for owner in self.machine_ids
            ],
        ]

    def _normal_on(self, step, owner):
        machine = self.machines_by_id[owner]
        mode = next(
            mode for mode in machine.quality_modes if mode.quality_mode_id == "normal"
        )
        return nominal_ticks(step, machine, mode, self.settings.rounding)

    def _decision(self, request, view, selection):
        _, count = self._scales(selection)
        state = view["agvs"].get(request.owner, {})
        divisor = count if self.global_unlimited else max(1, self.global_capacity)
        occupied = sum(
            row["location"] in self.global_owners for row in view["jobs"].values()
        )
        return [
            [
                occupied / divisor,
                float(self.global_unlimited),
                *[
                    float(request.role == role)
                    for role in ("machine", "buffer", "dispatcher", "mover")
                ],
                request.count / count,
                len(request.prefix) / count,
                float(state.get("job") is not None),
                float(bool(state.get("service"))),
                float(bool(state.get("feedback"))),
            ]
        ]

    def _self_target(self, request, view, selection):
        state = view["agvs"].get(request.owner, {})
        target = state.get("target") or {}
        return [
            [
                float(bool(target)),
                float(state.get("job") is not None),
                *[float(owner == target.get("owner")) for owner in self.owners],
                *[float(port == target.get("port")) for port in self.port_ids],
            ]
        ]

    def _local_map(self, request, view, selection):
        if request.role not in ("dispatcher", "mover"):
            return [[0.0] * self.widths["local_map"]]
        if request.owner not in view["agvs"]:
            raise ValueError("local map requires an actual owner AGV")
        location = tuple(view["agvs"].get(request.owner, {}).get("cell", (0, 0)))
        occupied = {tuple(state["cell"]) for state in view["agvs"].values()}
        values = [1.0]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = location[0] + dx, location[1] + dy
                outside = not (
                    0 <= cell[0] < self.factory.grid.width
                    and 0 <= cell[1] < self.factory.grid.height
                )
                values.extend(
                    (
                        float(outside or cell in self.solids),
                        float(cell in occupied and cell != location),
                    )
                )
        return [values]

    def _vehicles(self, request, view, selection):
        normalized = selection.normalization == "projection"
        return [
            [
                state["cell"][0] / (self.factory.grid.width if normalized else 1),
                state["cell"][1] / (self.factory.grid.height if normalized else 1),
                float(state["job"] is not None),
                float(state["service"]),
            ]
            for state in (view["agvs"][owner] for owner in self.agv_ids)
        ]

    def _machines(self, request, view, selection):
        time, _ = self._scales(selection)
        return [
            [
                float(state["down"]),
                *[
                    float(state["status"] == status)
                    for status in ("IDLE", "READY", "PROCESSING", "BLOCKED")
                ],
                state["nominal"] / time if state["job"] is not None else 0.0,
                state["elapsed"] / time if state["job"] is not None else 0.0,
                float(state["job"] is not None),
            ]
            for state in (view["machines"][owner] for owner in self.machine_ids)
        ]

    def _inventory(self, request, view, selection):
        result = []
        for owner in self.owners:
            rows = [
                (job, row)
                for job, row in view["jobs"].items()
                if row["location"] == owner and owner not in self.outputs
            ]
            supply = view["sources"].get(owner, {})
            active_ready = [
                job for job in supply.get("ready", ()) if job not in view["disposing"]
            ]

            def ratio(value):
                return self._ratio(value, owner, selection)

            capacity = self.capacities[owner]
            result.append(
                [
                    ratio(len(rows)),
                    ratio(len(active_ready)),
                    ratio(supply.get("supply", 0)),
                    (capacity or 0) / self._scales(selection)[1],
                    float(capacity is None),
                    *[
                        ratio(sum(row["quality"] == quality for _, row in rows))
                        for quality in ("PASS", "UNKNOWN", "FAIL")
                    ],
                    ratio(sum(job in view["disposing"] for job, _ in rows)),
                    *[
                        ratio(
                            sum(
                                row["next_operation"] == kind
                                and job not in view["disposing"]
                                for job, row in rows
                            )
                        )
                        for kind in self.operation_types
                    ],
                ]
            )
        return result

    def _topology(self, request, view, selection):
        return [self.map]

    def _work_summary(self, request, view, selection):
        time, count = self._scales(selection)
        rows = [
            self.work(job, view, time)
            for job, row in view["jobs"].items()
            if row["location"] in self.physical and job not in view["disposing"]
        ]
        if not rows:
            return [[0.0] * len(CRITIC_FIELDS)]
        urgency, fraction, remaining = np.asarray(rows).T
        divisor = count if self.global_unlimited else max(1, self.global_capacity)
        return [
            [
                urgency.min(),
                urgency.mean(),
                (urgency < 0).mean(),
                remaining.mean(),
                remaining.max(),
                remaining.sum() / divisor,
                fraction.mean(),
                1.0,
            ]
        ]

    def _context(self, request, view):
        # Compute each selected component once. No equal-width zero planes are
        # constructed, copied into packets or retained in DQN replay.
        values_by_name = {}
        for selection in self.settings.context:
            if self.algorithm == "dqn" and selection.visibility == "critic":
                continue
            rows = np.asarray(
                CONTEXT_ENCODERS[selection.name](self, request, view, selection),
                dtype=np.float32,
            ).reshape(-1, self.widths[selection.name])
            if selection.aggregation == "per_entity":
                values = rows.reshape(-1)
            elif len(rows):
                values = getattr(rows, selection.aggregation)(axis=0)
            else:
                values = np.zeros(self.widths[selection.name], np.float32)
            if (
                selection.applies_to is not None
                and request.role not in selection.applies_to
            ):
                values = np.zeros_like(values)
            values_by_name[selection.name] = values
        return (
            np.concatenate(
                [values_by_name[row["component"]["name"]] for row in self.layout]
            ).astype(np.float32, copy=False)
            if self.layout
            else np.empty(0, np.float32)
        )

    def _candidate(self, request, view, selection, candidate=None, prefix_job=None):
        if (
            selection.applies_to is not None
            and request.role not in selection.applies_to
        ):
            return [0.0] * self.widths[selection.name]
        role = "buffer" if prefix_job is not None else request.role
        job = (
            prefix_job
            if prefix_job is not None
            else candidate.action[0]
            if role == "machine"
            else candidate.action
            if role == "buffer"
            else None
        )
        cargo = view["agvs"].get(request.owner, {}).get("job")
        time, _ = self._scales(selection)
        if selection.name == "job":
            return self._job(job, view, selection)
        if selection.name == "cargo":
            return (
                self._job(cargo, view, selection)
                if role == "dispatcher"
                else [0.0] * len(JOB_FIELDS)
            )
        if selection.name == "semantics":
            return self._semantics(
                cargo if role == "dispatcher" else job,
                view,
                selection,
                target=candidate.action if role == "dispatcher" else None,
                after_start=role == "machine",
            )
        if selection.name == "machine_mode" and role == "machine":
            machine = self.machines_by_id[request.owner]
            mode = next(
                mode
                for mode in machine.quality_modes
                if mode.quality_mode_id == candidate.action[1]
            )
            probability = candidate.features[10]
            return [
                nominal_ticks(
                    view["jobs"][job]["remaining_steps"][0],
                    machine,
                    mode,
                    self.settings.rounding,
                )
                / time,
                float(mode.time_scale),
                max(0, probability),
                float(probability >= 0),
                float(mode.quality_mode_id == "normal"),
            ]
        if selection.name == "dispatch_target" and role == "dispatcher":
            target = candidate.action
            rows = [
                row
                for row in view["jobs"].values()
                if row["location"] == target.owner and target.owner not in self.outputs
            ]
            capacity = self.capacities[target.owner]
            supply = view["sources"].get(target.owner, {})

            def ratio(value):
                return self._ratio(value, target.owner, selection)

            own_target = view["agvs"][request.owner].get("target") or {}
            return [
                1.0,
                candidate.features[1]
                * request.observation.get("feature_time_scale", 100)
                / time,
                candidate.features[2],
                candidate.features[3],
                float(cargo is not None),
                ratio(supply.get("supply", 0)),
                ratio(len(rows)),
                (capacity or 0) / self.projection["count_scale"],
                float(capacity is None),
                float(capacity is not None and len(rows) >= capacity),
                float(own_target == {"owner": target.owner, "port": target.port}),
                ratio(
                    len(
                        [
                            job
                            for job in supply.get("ready", ())
                            if job not in view["disposing"]
                        ]
                    )
                ),
            ]
        if selection.name == "movement" and role == "mover":
            values = list(candidate.features)
            values[2] *= request.observation.get("feature_time_scale", 100) / time
            return values
        return [0.0] * self.widths[selection.name]

    def encode(self, request):
        view = restricted_view(request.observation, request.owner)
        # Shipped history is not physical resident inventory or available work.
        view["jobs"] = {
            job: row
            for job, row in view["jobs"].items()
            if row["location"] in self.physical
        }
        if any(
            row["quality"] == "FAIL" and job not in view["disposing"]
            for job, row in view["jobs"].items()
        ):
            raise ValueError(
                "public FAIL violates automatic-disposal observation invariant"
            )
        # Components receive this restricted request too. The old candidate tuple
        # contained intention counts and accumulated Output inventory.
        candidates = []
        for candidate in request.candidates:
            values = list(candidate.features)
            if request.role == "dispatcher":
                values[5:11] = [0.0] * 6
            elif request.role in ("machine", "buffer"):
                values[4] = 0.0  # historical attempt scalar is metrics-only
            candidates.append(replace(candidate, features=tuple(values)))
        request = replace(request, observation=view, candidates=tuple(candidates))
        candidates = [
            [
                value
                for selection in self.settings.candidates
                for value in self._candidate(request, view, selection, candidate)
            ]
            for candidate in request.candidates
        ]
        prefix = [
            [
                value
                for selection in self.settings.candidates
                for value in self._candidate(request, view, selection, prefix_job=job)
            ]
            for job in request.prefix
        ]
        result = {
            "context": self._context(request, view),
            "candidates": np.asarray(candidates, np.float32).reshape(
                -1, self.candidate_width
            ),
            "prefix": np.asarray(prefix, np.float32).reshape(-1, self.candidate_width),
            "mask": np.asarray(
                [candidate.legal for candidate in request.candidates], bool
            ),
            "feature_width": np.asarray(self.candidate_width),
        }
        if len(result["context"]) != self.context_size or not all(
            np.isfinite(value).all() for value in result.values()
        ):
            raise ValueError("invalid registered public observation shape/value")
        return result


# Extend this explicit registry in Python for new computations. YAML can select
# only these names and the typed operations declared by the configuration registry.
CONTEXT_ENCODERS = {
    name: getattr(PublicFieldsEncoder, "_" + name)
    for name in (
        "decision",
        "self_target",
        "local_map",
        "vehicles",
        "machines",
        "inventory",
        "topology",
        "work_summary",
    )
}


class ScopedCandidateNetwork(CandidateNetwork):
    DIAGNOSTIC_CONTRACT = {
        **CandidateNetwork.DIAGNOSTIC_CONTRACT,
        "schema": "smartsom.scoped-candidate/v2",
        "context_planes": ["compact actor/Q only", "compact critic only"],
        "critic_prefix": "absent",
        "critic_candidate_pool": "absent",
    }

    def __init__(
        self,
        context_size,
        network,
        provider,
        algorithm="ppo",
        *,
        context_layout,
        central=False,
        central_private_end=0,
        candidate_width=16,
    ):
        nn.Module.__init__(self)
        if (
            set(context_layout) != {"actor", "critic"}
            or any(
                type(value) is not int or value < 0 for value in context_layout.values()
            )
            or sum(context_layout.values()) != context_size
        ):
            raise ValueError("compact context layout mismatch")
        if algorithm not in ("ppo", "dqn") or (
            algorithm == "dqn" and context_layout["critic"]
        ):
            raise ValueError("compact DQN must not include critic context")
        self.context_size, self.algorithm, self.central = (
            context_size,
            algorithm,
            central,
        )
        self.central_private_end = central_private_end
        self.candidate_width = candidate_width
        self.actor_context_size = context_layout["actor"]
        self.critic_context_size = context_layout["critic"]
        self.actor_prefix = nn.GRU(candidate_width, PREFIX_WIDTH, batch_first=True)
        actor_space = ObservationSpace(
            vector=(self.actor_context_size + PREFIX_WIDTH + candidate_width,)
        )
        if algorithm == "dqn":
            selected = QNetwork.model_validate_json(json.dumps(network))
            self.actor = _Branch(actor_space, 1, selected.q, provider)
            self.critic = None
        else:
            selected = ActorCriticSpec.model_validate_json(json.dumps(network))
            self.actor = _Branch(actor_space, 1, selected.actor, provider)
            self.critic = _Branch(
                ObservationSpace(vector=(self.critic_context_size,)),
                1,
                selected.critic,
                provider,
            )

    def forward(self, observations):
        context = observations["context"].float()
        if context.shape[-1] != self.context_size:
            raise ValueError("compact context shape mismatch")
        actor = context[:, : self.actor_context_size]
        prefix = self.prefix_embedding(
            observations["prefix"].float(),
            observations["prefix_length"].long(),
            self.actor_prefix,
        )
        candidates = observations["candidates"].float()
        expanded = (
            torch.cat((actor, prefix), -1)
            .unsqueeze(1)
            .expand(
                -1,
                candidates.shape[1],
                -1,
            )
        )
        scores = self.actor(torch.cat((expanded, candidates), -1)).squeeze(-1)
        scores = scores.masked_fill(
            ~observations["mask"].bool(),
            torch.finfo(scores.dtype).min,
        )
        if self.critic is None:
            return scores, scores.max(-1).values
        return scores, self.critic(context[:, self.actor_context_size :]).squeeze(-1)
