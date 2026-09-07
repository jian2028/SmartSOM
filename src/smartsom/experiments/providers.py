"""Explicit construction of supported providers at the experiment boundary."""

from smartsom.algorithms import FirstFeasiblePolicy, ScriptedPolicy, SPTPolicy
from smartsom.algorithms.pyjobshop import PyJobShopAdapter
from smartsom.algorithms.solver import SolverAdapter
from smartsom.config.models import AlgorithmFile, CPSatAlgorithm, ScriptedAlgorithm
from smartsom.dispatch import OnlinePolicy


def build_provider(algorithm: AlgorithmFile) -> OnlinePolicy | SolverAdapter:
    spec = algorithm.algorithm
    if isinstance(spec, ScriptedAlgorithm):
        return ScriptedPolicy(spec.parameters.actions)
    if spec.provider == "builtin.spt":
        return SPTPolicy()
    if spec.provider == "builtin.first_feasible":
        return FirstFeasiblePolicy()
    if isinstance(spec, CPSatAlgorithm):
        return PyJobShopAdapter()
    raise ValueError(f"unsupported provider: {spec.provider!r}")
