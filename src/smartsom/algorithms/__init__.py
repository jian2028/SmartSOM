"""Built-in policies over the current detached grid decision contract."""

from smartsom.algorithms.production import (
    GreedyProductionPolicy,
    RandomProductionPolicy,
    ScriptedProductionPolicy,
)


def SPTPolicy(factory, quality_mode=None, *, seed=0):
    """Shortest nominal processing time among the machine's eligible choices."""
    return GreedyProductionPolicy(factory, seed, rule="spt", quality_mode=quality_mode)


def FirstFeasiblePolicy(factory, quality_mode=None, *, seed=0):
    """Select by stable semantic identity, with explicit grid transport."""
    return GreedyProductionPolicy(
        factory, seed, rule="first_feasible", quality_mode=quality_mode
    )


ScriptedPolicy = ScriptedProductionPolicy

__all__ = [
    "FirstFeasiblePolicy",
    "ScriptedPolicy",
    "SPTPolicy",
    "GreedyProductionPolicy",
    "RandomProductionPolicy",
]
