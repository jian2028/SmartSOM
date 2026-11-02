"""Potential-based AGV target-distance reward shaping (Huang & Jin 2022 form).

Each vehicle with a public target contributes exp(-gradient * d / cells), where d
is its static grid path distance to the target port. The potential is the fleet
mean scaled by weight, and the shaping term is gamma * phi(after) - phi(before),
with phi(after) = 0 when the episode ends. This difference form leaves optimal
policies unchanged when gamma equals the learner's discount.
"""

from collections import deque
from math import exp, isfinite
from numbers import Real

from smartsom.learning.extensions import register_extension

PARAMETERS = {"weight", "gradient", "cells", "gamma"}


class AgvDistancePotential:
    def __init__(self, parameters):
        if set(parameters) != PARAMETERS:
            raise ValueError(
                "builtin.agv_distance_potential requires weight, gradient, cells and gamma"
            )
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(value)
            for value in parameters.values()
        ):
            raise ValueError("reward shaping parameters must be finite numbers")
        self.weight = float(parameters["weight"])
        self.gradient = float(parameters["gradient"])
        self.cells = float(parameters["cells"])
        self.gamma = float(parameters["gamma"])
        if self.weight < 0 or self.gradient <= 0 or self.cells <= 0:
            raise ValueError("weight must be nonnegative; gradient and cells positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("shaping gamma must be in (0, 1]")
        self._distances = {}

    def potential(self, view):
        vehicles = view["agvs"]
        if not vehicles:
            return 0.0
        total = 0.0
        for state in vehicles.values():
            if not state.get("target"):
                continue
            distance = self.distances(view, state["target"]["port"]).get(
                tuple(state["cell"])
            )
            if distance is not None:
                total += exp(-self.gradient * distance / self.cells)
        return self.weight * total / len(vehicles)

    def distances(self, view, port):
        """Static breadth-first path lengths to a port; vehicles never block it."""
        if port not in self._distances:
            topology = view["topology"]
            solids = {tuple(cell) for cell in topology["solids"]}
            target = tuple(topology["ports"][port])
            distances, queue = {target: 0}, deque([target])
            while queue:
                x, y = queue.popleft()
                for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (
                        0 <= nxt[0] < topology["width"]
                        and 0 <= nxt[1] < topology["height"]
                        and nxt not in solids
                        and nxt not in distances
                    ):
                        distances[nxt] = distances[(x, y)] + 1
                        queue.append(nxt)
            self._distances[port] = distances
        return self._distances[port]

    def transform(self, transition, reward):
        after = (
            0.0 if transition.reason is not None else self.potential(transition.after)
        )
        return reward + self.gamma * after - self.potential(transition.before)


register_extension(
    "reward", "builtin.agv_distance_potential", "1", AgvDistancePotential
)
