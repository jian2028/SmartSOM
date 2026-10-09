"""Passive phase-specific progress accounting and bounded event context.

Co-occurring conditions are evidence for analysis, not a causal attribution.
Never install these counters in a simulator or a public observation.
"""

import json

from smartsom.config.codec import digest, primitive
from smartsom.config.diagnostics import EventContextOptions

EVENT_CAP = 512 * 1024
EXAMPLE_LIMIT = 8
STAGNATION_TICKS = 128
DEFAULT_EVENT_CONTEXT = primitive(EventContextOptions())


def initial_state(event_context=None):
    settings = primitive(EventContextOptions.model_validate(event_context or {}))
    return {
        "capture_settings": settings,
        "capture_sha256": digest(settings),
        "capture_changes": [],
        "capture_changes_omitted": 0,
        "schema": "smartsom.progress-diagnostics/v1",
        "boundaries": 0,
        "counts": {},
        "progress": {},
        "recent": [],
        "snippets": [],
        "snippet_bytes": 0,
        "suppressed_triggers": 0,
        "history_complete": True,
    }


def configure_capture(state, event_context, tick):
    """Begin an explicit retention segment without rewriting historical counters."""
    settings = primitive(EventContextOptions.model_validate(event_context))
    previous = state.get("capture_settings", DEFAULT_EVENT_CONTEXT)
    if previous != settings:
        changes = state.setdefault("capture_changes", [])
        changes.append(
            {
                "effective_from_tick": tick,
                "previous_sha256": digest(previous),
                "capture_sha256": digest(settings),
                "discarded_snippets": len(state["snippets"]),
                "discarded_recent_boundaries": len(state["recent"]),
            }
        )
        state["capture_changes_omitted"] = state.get(
            "capture_changes_omitted", 0
        ) + max(0, len(changes) - 8)
        state["capture_changes"] = changes[-8:]
        state["snippets"], state["recent"], state["snippet_bytes"] = [], [], 0
        state["capture_started_tick"] = tick
        state["context_history_complete"] = False
    state["capture_settings"] = settings
    state["capture_sha256"] = digest(settings)


def _count(state, key, condition, applicable=True):
    row = state["counts"].setdefault(
        key, {"numerator": 0, "denominator": 0, "unknown": 0, "not_applicable": 0}
    )
    row["denominator"] += int(applicable is True)
    row["numerator"] += int(bool(condition) and applicable is True)
    row["unknown"] += int(applicable is None)
    row["not_applicable"] += int(applicable is False)


def _compact(values, keys, limit=8):
    return [
        {
            k: v[:64] if isinstance(v, str) else v
            for k, v in row.items()
            if k in keys and (v is None or isinstance(v, (str, int, float, bool)))
        }
        for row in values[:limit]
    ]


def destination_full(sim, view, action):
    """Proposal-phase capacity at the selected port, before simultaneous admission."""
    owner = action["owner"]
    port = sim.protocol.ports.get(action["port"])
    if port is None:
        return None
    if owner in sim.scrap:
        cap = sim.scrap[owner].capacity
        if cap is None:
            return False
        used = view.get("metrics", {}).get("scrap:" + owner, 0)
        return used >= cap
    if owner in sim.machines:
        machine = view.get("machines", {}).get(owner)
        return machine["job"] is not None if machine is not None else None
    storage = view.get("storage", {}).get(owner)
    capacity = sim.capacity.get(owner)
    if storage is None or capacity is None:
        return None
    allowed = {
        sim._target(b.target)[1]
        for b in port.bindings
        if "drop_off" in b.operations and sim._target(b.target)[0] == owner
    }
    for slot, jobs in storage.items():
        if slot not in allowed:
            continue
        if slot not in capacity:
            return None
        cap = capacity[slot]
        if cap is None or len(jobs) < cap:
            return False
    return True


def seed_restored_progress(state, sim):
    """Supply the pre-transition baseline when an old checkpoint lacks counters."""
    state.setdefault("last_shipped", len(sim.shipped))
    state.setdefault("observed_start_tick", sim.tick)


def observe(state, coordinator, outcome):
    sim = coordinator.sim
    settings = state.get("capture_settings", DEFAULT_EVENT_CONTEXT)
    state["boundaries"] += 1
    tick = sim.tick
    state.setdefault("observed_start_tick", max(0, tick - 1))
    previous_shipped = state.get(
        "last_shipped", 0 if state.get("history_complete", False) else len(sim.shipped)
    )
    for record in coordinator.records:
        if record["role"] != "dispatcher":
            continue
        view = record["observation"]
        vehicle = view["agvs"][record["owner"]]
        action = record["proposal"]
        if not isinstance(action, dict):
            continue
        empty = vehicle["job"] is None
        source = view.get("sources", {}).get(action["owner"])
        _count(
            state,
            "dispatch_empty_selected_source_no_ready_stock",
            not source.get("ready") if source else False,
            False if not empty else True if source is not None else None,
        )
        if not empty:
            full = destination_full(sim, view, action)
            _count(
                state,
                "dispatch_loaded_selected_destination_full",
                full,
                True if full is not None else None,
            )
        _count(
            state,
            "dispatch_retarget_existing_intention",
            vehicle.get("target") != action,
            vehicle.get("target") is not None,
        )
    for vehicle in sim.agvs.values():
        _count(
            state,
            "postcommit_agv_travelling",
            bool(vehicle.get("travel")),
            True if "travel" in vehicle else None,
        )
        _count(state, "postcommit_agv_in_service", bool(vehicle.get("service")))
        target = vehicle.get("target")
        if target and not vehicle.get("travel") and not vehicle.get("service"):
            port = sim.protocol.ports.get(target["port"])
            arrived = port is not None and tuple(vehicle["cell"]) == (
                port.cell.x,
                port.cell.y,
            )
            if arrived:
                competing = (
                    sum(
                        bool(
                            other.get("target")
                            and other["target"]["port"] == target["port"]
                            and tuple(other["cell"]) == tuple(vehicle["cell"])
                        )
                        for other in sim.agvs.values()
                    )
                    > 1
                )
                _count(
                    state, "postcommit_arrived_idle_agv_shared_target_port", competing
                )
    for machine in sim.machine_state.values():
        _count(
            state, "postcommit_machine_processing", machine["status"] == "PROCESSING"
        )
    kinds = [event["kind"] for event in outcome.get("events", ())]
    signals = {
        "pickup": kinds.count("pickup_started"),
        "operation": kinds.count("processing_completed"),
        "shipment": max(0, len(sim.shipped) - previous_shipped),
    }
    state["last_shipped"] = len(sim.shipped)
    triggers = []
    for name, count in signals.items():
        row = state["progress"].setdefault(
            name, {"events": 0, "last_tick": None, "max_observed_gap": 0}
        )
        start = (
            row["last_tick"]
            if row["last_tick"] is not None
            else state["observed_start_tick"]
        )
        gap = tick - start
        row["max_observed_gap"] = max(row["max_observed_gap"], gap)
        row["events"] += count
        if count:
            row["last_tick"] = tick
        row["open_gap_ticks"] = 0 if count else gap
        row["right_censored"] = not bool(count)
        trigger_gap = tick - max(start, state.get("capture_started_tick", start))
        if not count and trigger_gap == settings["stagnation_ticks"]:
            triggers.append(name + f"_no_progress_{settings['stagnation_ticks']}_ticks")
    if outcome.get("rejections"):
        triggers.append("committed_rejection")
    compact = {
        "tick": tick,
        "decisions": _compact(
            coordinator.records,
            {"tick", "stage", "role", "owner", "candidate", "group"},
        ),
        "events": _compact(
            outcome.get("events", []),
            {"tick", "kind", "agv", "job", "owner", "machine"},
        ),
        "decisions_omitted": max(0, len(coordinator.records) - 8),
        "events_omitted": max(0, len(outcome.get("events", [])) - 8),
    }
    state["recent"].append(compact)
    state["recent"] = state["recent"][-settings["lookback_boundaries"] :]
    if triggers:
        snippet = {
            "triggers": triggers,
            "context": list(state["recent"]),
            "interpretation": "ordered event context; not proof of causal blame",
        }
        size = len(json.dumps(snippet, ensure_ascii=False).encode())
        if (
            len(state["snippets"]) < settings["max_snippets"]
            and state["snippet_bytes"] + size <= settings["max_payload_bytes"]
        ):
            state["snippets"].append(snippet)
            state["snippet_bytes"] += size
        else:
            state["suppressed_triggers"] += 1


def summary(state):
    import copy

    result = copy.deepcopy(state)
    result.pop("recent", None)
    settings = state.get("capture_settings", DEFAULT_EVENT_CONTEXT)
    result.update(
        scope="one physical episode, aggregate owners; conditions may overlap",
        phase="dispatch proposal and post-commit owner-boundaries; progress uses committed events",
        snippet_payload_cap_bytes=settings["max_payload_bytes"],
        example_limit=settings["max_snippets"],
        stagnation_threshold_ticks=settings["stagnation_ticks"],
    )
    return result
