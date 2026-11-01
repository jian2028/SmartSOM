"""Decode frozen physical scenarios for read-only evidence and playback."""

import json

from pydantic import ConfigDict, TypeAdapter

from smartsom.domain.production import ProductionScenario


def scenario_from_snapshot(data):
    return TypeAdapter(
        tuple[ProductionScenario, ...], config=ConfigDict(extra="forbid")
    ).validate_json(json.dumps([normalize_duration_tables(data)]), strict=True)[0]


def normalize_duration_tables(data):
    """Decode YAML/JSON mappings into the immutable domain representation."""
    import copy

    data = copy.deepcopy(data)
    for demand in data.get("demands") or ():
        if not isinstance(demand, dict):
            continue
        for step in demand.get("steps", ()):
            if not isinstance(step, dict):
                continue
            table = step.get("machine_nominal_ticks", ())
            if isinstance(table, dict):
                step["machine_nominal_ticks"] = list(table.items())
    return data
