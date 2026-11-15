"""Fixed numeric V2 observations, with semantic actions kept outside tensors.

Bounds describe the frozen task, not measured resource usage. Overflow is an
error: neither candidates nor social relations may be silently discarded.
"""

import gymnasium as gym
import numpy as np

from smartsom.representations.v2.observation import CONTEXT, FEATURES
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import social_config

SUPPLY_FIELDS = frozenset(
    {"jobs", "job_mask", "job_peers", "job_peer_mask", "job_meta", "job_count"}
)


class V2ObservationCodec:
    def __init__(self, group, *, options, actions, stations, vehicles, jobs, social):
        if any(
            type(n) is not int or n < 1 for n in (actions, stations, vehicles, jobs)
        ):
            raise ValueError("observation bounds must be positive integers")
        self.group, self.options = group, options
        self.settings = social_config(social)
        role, _, selected_stage = group.partition(".")
        self.stages = tuple(
            point.split(".")[1]
            for point in CATALOG
            if point.startswith(role + ".")
            and (not selected_stage or point.endswith("." + selected_stage))
        )
        if not self.stages or (role == "dispatcher" and not selected_stage):
            raise ValueError(
                "V2 codec requires a supported group and split dispatcher stages"
            )
        for stage in self.stages:
            point = role + "." + stage
            name = options.production.bindings.get(point, options.production.default)
            policy = options.production.policies[name]
            for spec in (policy.observation, policy.network):
                if spec is None or social_config(**spec.params) != self.settings:
                    raise ValueError(
                        "codec and observation/network social configuration mismatch"
                    )
        self.width = actions + 1  # Zero is reserved for the real learned WAIT.
        self.shapes = {}
        slots = self.settings.slots
        stations = slots.stations if slots else stations
        vehicles = slots.vehicles if slots else vehicles
        for stage in self.stages:
            shapes = {
                "context": (CONTEXT.get(group, CONTEXT.get(role)),),
                "candidates": (
                    self.width,
                    FEATURES[role + "." + stage] + len(CATALOG[role + "." + stage]),
                ),
                "modes": (self.width, 3),
                "mask": (self.width,),
            }
            if self.settings.kind == "implicit":
                if role == "dispatcher":
                    shapes.update(
                        peers=(self.width, vehicles, 3),
                        peer_mask=(self.width, vehicles),
                    )
                elif role in ("input", "post") and stage == "accept":
                    count = (
                        (slots.input_jobs if role == "input" else slots.post_jobs)
                        if slots
                        else jobs
                    )
                    shapes.update(
                        # Acceptance supply is owner-specific, identical for
                        # every candidate vehicle. Store it once; the model
                        # broadcasts it back to the existing research network.
                        jobs=(count, 3 if role == "input" else 4),
                        job_mask=(count,),
                        job_peers=(count, stations, 9),
                        job_peer_mask=(count, stations),
                        job_meta=(count, 2),
                        job_count=(1,),
                    )
                else:
                    shapes.update(
                        downstream=(self.width, stations, 9),
                        downstream_mask=(self.width, stations),
                        downstream_meta=(self.width, 2),
                        downstream_count=(self.width, 1),
                    )
                    if role == "pre" and stage == "accept":
                        shapes.update(
                            alternatives=(self.width, stations, 9),
                            alternatives_mask=(self.width, stations),
                            alternatives_count=(self.width, 1),
                        )
            self.shapes[stage] = shapes
        self.observation_space = gym.spaces.Dict(
            {
                "stage": gym.spaces.Discrete(len(self.stages)),
                "waiting": gym.spaces.MultiBinary(1),
                "action_mask": gym.spaces.MultiBinary(self.width),
                "stages": gym.spaces.Dict(
                    {
                        stage: gym.spaces.Dict(
                            {
                                key: gym.spaces.MultiBinary(shape)
                                if key.endswith("mask")
                                else gym.spaces.Box(
                                    -np.inf, np.inf, shape=shape, dtype=np.float32
                                )
                                for key, shape in shapes.items()
                            }
                        )
                        for stage, shapes in self.shapes.items()
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.width)

    def _empty(self, stage, waiting):
        if stage not in self.stages:
            raise ValueError("decision stage does not belong to this module")
        return {
            "stage": self.stages.index(stage),
            "waiting": np.array([waiting], dtype=np.int8),
            "action_mask": np.zeros(self.width, dtype=np.int8),
            "stages": {
                name: {
                    key: np.zeros(
                        shape, dtype=np.int8 if key.endswith("mask") else np.float32
                    )
                    for key, shape in shapes.items()
                }
                for name, shapes in self.shapes.items()
            },
        }

    def encode(self, request):
        from smartsom.representations.v2.network import pack

        self._validate_request(request)
        if not request.candidates or len(request.candidates) >= self.width:
            raise ValueError("candidate count outside frozen action capacity")
        result = self._empty(request.stage, False)
        packed = pack([request], critics=None, options=self.options)["stages"][
            request.stage
        ]
        destination = result["stages"][request.stage]
        if set(packed) - {"indices"} != set(destination):
            raise ValueError("observation fields do not match the frozen social schema")
        for key, tensor in packed.items():
            if key == "indices":
                continue
            if key not in destination:
                raise ValueError(f"unexpected observation field: {key}")
            value = tensor[0].numpy()
            if key in SUPPLY_FIELDS:
                if not np.array_equal(value, np.broadcast_to(value[0], value.shape)):
                    raise ValueError("acceptance supply differs between candidates")
                value = value[0]
            target = (
                destination[key]
                if key == "context" or key in SUPPLY_FIELDS
                else destination[key][1:]
            )
            if value.ndim != target.ndim or any(
                a > b for a, b in zip(value.shape, target.shape, strict=True)
            ):
                raise ValueError(f"observation {key} exceeds frozen capacity")
            target[tuple(slice(0, n) for n in value.shape)] = value
        result["action_mask"][:] = destination["mask"]
        return result

    def encode_wait(self, protocol, context):
        """Use fresh permitted local context, never a cached production request."""
        from smartsom.engine.decision_facts import DecisionFacts
        from smartsom.representations.v2.facts import view
        from smartsom.representations.v2.observation import context as local_context

        self._validate_request(context)
        result = self._empty(context.stage, True)
        with protocol.local_facts(), DecisionFacts(protocol, context) as facts:
            public = view(facts)
            if (
                context.group == "dispatcher"
                and context.stage == "delivery"
                and public.f.agvs[context.owner]["job"] is None
            ):
                values = (0.0, 0.0, 0.0)
            else:
                values = local_context(public, context)
        result["stages"][context.stage]["context"][:] = values
        result["stages"][context.stage]["mask"][0] = 1
        result["action_mask"][0] = 1
        return result

    def decode(self, request, action):
        self._validate_request(request)
        if (
            not self.action_space.contains(action)
            or action == 0
            or action > len(request.candidates)
        ):
            raise ValueError("action index is not a legal production candidate")
        return request.candidates[int(action) - 1].identity

    def _validate_request(self, request):
        if (
            request.group != self.group.split(".")[0]
            or request.stage not in self.stages
        ):
            raise ValueError("decision role/stage does not belong to this module")
