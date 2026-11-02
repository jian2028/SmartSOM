"""Potential-based production shaping: AGV target distance plus job progress.

phi = mean over vehicles of w_v * exp(-gradient * d / cells), with w_v = weight
for loaded vehicles and empty_weight for empty ones, plus progress_weight *
(shipped demands - remaining operations of released, unshipped attempts).
The shaping term is gamma * phi(after) - phi(before), with phi(after) = 0 at the
episode end, so optimal policies are unchanged when gamma matches the learner.
"""

from math import exp, isfinite
from numbers import Real
from pathlib import Path

from smartsom.learning.extensions import register_extension
from smartsom.learning.reward_shaping import AgvDistancePotential

PARAMETERS = {"weight", "empty_weight", "gradient", "cells", "gamma", "progress_weight"}


class ProductionPotential(AgvDistancePotential):
    def __init__(self, parameters):
        if set(parameters) != PARAMETERS:
            raise ValueError(
                "builtin.production_potential requires " + ", ".join(sorted(PARAMETERS))
            )
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(value)
            for value in parameters.values()
        ):
            raise ValueError("reward shaping parameters must be finite numbers")
        super().__init__(
            {key: parameters[key] for key in ("weight", "gradient", "cells", "gamma")}
        )
        self.empty_weight = float(parameters["empty_weight"])
        self.progress_weight = float(parameters["progress_weight"])
        if self.empty_weight < 0 or self.progress_weight < 0:
            raise ValueError("empty_weight and progress_weight must be nonnegative")

    def potential(self, view):
        vehicles = view["agvs"]
        distance = 0.0
        for state in vehicles.values():
            if not state.get("target"):
                continue
            steps = self.distances(view, state["target"]["port"]).get(
                tuple(state["cell"])
            )
            if steps is not None:
                scale = (
                    self.weight if state.get("job") is not None else self.empty_weight
                )
                distance += scale * exp(-self.gradient * steps / self.cells)
        distance = distance / len(vehicles) if vehicles else 0.0
        shipped = set(view.get("shipped", ()))
        remaining = sum(
            len(job.get("remaining_steps", ()))
            for job in view.get("jobs", {}).values()
            if job.get("demand") not in shipped
        )
        return distance + self.progress_weight * (len(shipped) - remaining)


register_extension(
    "reward",
    "builtin.production_potential",
    "1",
    ProductionPotential,
    source_files=(Path(__file__).with_name("reward_shaping.py"),),
)
