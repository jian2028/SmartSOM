"""Small built-in policies using the existing online decision contract."""

from smartsom.algorithms.spt import SPTPolicy
from smartsom.algorithms.toy import FirstFeasiblePolicy, ScriptedPolicy

__all__ = ["FirstFeasiblePolicy", "ScriptedPolicy", "SPTPolicy"]
