"""Execution preflight, separate from learner optimization and calibration."""

import json
import multiprocessing
import time
from pathlib import Path

from smartsom.experiments.control import boundary, write_json
from smartsom.telemetry.runtime import CURRENT

SCHEMA = "smartsom.preflight/v1"
SMOKE_TICKS = 128
SMOKE_TIMEOUT = 120.0


def _smoke_child(prepared, result):
    """Run disposable policy decisions; never create a learner update."""
    try:
        from smartsom.algorithms.production_composition import BoundaryCoordinator
        from smartsom.config.production import scenario_from_snapshot
        from smartsom.engine.production import ProductionSimulator
        from smartsom.experiments.composable import bindings, policies_for

        policies, learners = policies_for(prepared, training=True)
        cases = json.loads(prepared.validation_json)
        scenario = (
            scenario_from_snapshot(cases[0]["scenario"]) if cases else prepared.scenario
        )
        sim = ProductionSimulator(scenario, contract="v3")
        matching = json.loads(prepared.composition_json)["matching"]["name"]
        coordinator = BoundaryCoordinator(sim, policies, bindings(prepared), matching)
        ticks = 0
        touched = set()
        while ticks < SMOKE_TICKS and not sim.done:
            coordinator.tick()
            touched.update(record["group"] for record in coordinator.records)
            ticks += 1
            if touched >= set(policies):
                break
        result.send(
            {
                "status": "passed",
                "ticks": ticks,
                "exercised": sorted(touched),
                "uncovered": sorted(set(policies) - touched),
                "learner_groups": sorted(learners),
            }
        )
    except BaseException as exc:
        result.send({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        result.close()


def smoke(prepared, *, timeout=SMOKE_TIMEOUT, cancelled=lambda: False):
    reader, writer = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_smoke_child, args=(prepared, writer)
    )
    process.start()
    writer.close()
    try:
        deadline = time.monotonic() + timeout
        while not reader.poll(0.1):
            if cancelled():
                return {"status": "skipped"}
            if time.monotonic() >= deadline:
                return {"status": "failed", "error": "smoke exceeded 120s"}
        answer = reader.recv()
        process.join(5)
        if process.exitcode not in (0, None):
            return {"status": "failed", "error": "smoke worker exited abnormally"}
        return answer
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
            process.join()


def _representatives(entries):
    selected = set()
    seen = set()
    for entry in entries:
        key = json.dumps(
            [
                entry["sources"],
                entry["H"],
                entry["workload"].get("selected_level"),
                entry["task"],
            ],
            sort_keys=True,
            default=str,
        )
        if key not in seen:
            seen.add(key)
            selected.add(entry["id"])
    return selected


def _coverage(root, current):
    command = root / "control/preflight.json"
    if not command.is_file():
        return current
    from smartsom.experiments.control import read

    owner = read(root)
    requested = json.loads(command.read_text())
    if not owner or requested.get("owner_id") != owner["id"]:
        return current
    next_value = requested.get("coverage")
    if next_value not in {"representative", "skip"}:
        return current
    if current == "each" or current == "representative" and next_value == "skip":
        return next_value
    return current


def run(root, plan):
    from smartsom.experiments.composable import prepared_from_run, verify_prepared_rules

    root = Path(root)
    saved = root / "preflight.json"
    if saved.is_file():
        previous = json.loads(saved.read_text())
        if previous.get("status") == "passed":
            if CURRENT.get():
                CURRENT.get().configure_preflight(previous)
            return previous
    if plan.get("schema") == "smartsom.tune-batch/v1":
        settings = plan
        entries = [
            {
                "id": e["experiment_id"],
                "snapshot": f"inputs/{e['experiment_id']}",
                "sources": e["prepared"].get("sources", {}),
                "H": {},
                "workload": {},
                "task": "train-evaluate",
            }
            for e in plan["entries"]
        ]
    else:
        settings = plan["experiment"]["execution"]
        entries = plan["entries"]
    state = {
        "schema": SCHEMA,
        "status": "running",
        "level": settings.get("preflight", "quick"),
        "coverage": settings.get("preflight_coverage", "each"),
        "checks_total": len(entries) + 1,
        "checks_done": 1,  # author compilation has already succeeded
        "smoke_total": len(entries) if settings.get("preflight") == "full" else 0,
        "smoke_done": 0,
        "smoke_skipped": 0,
        "started_at": time.time(),
        "entries": {},
    }

    def publish():
        write_json(saved, state)
        if CURRENT.get():
            CURRENT.get().configure_preflight(state)

    publish()
    prepared = {}
    try:
        for entry in entries:
            boundary(root)
            value = prepared_from_run(root / entry["snapshot"])
            verify_prepared_rules(value)
            prepared[entry["id"]] = value
            state["checks_done"] += 1
            state["current"] = entry["id"]
            publish()
        if state["level"] == "full":
            representatives = _representatives(entries)
            for entry in entries:
                boundary(root)
                state["coverage"] = _coverage(root, state["coverage"])
                identity = entry["id"]
                if state["coverage"] == "skip" or (
                    state["coverage"] == "representative"
                    and identity not in representatives
                ):
                    state["entries"][identity] = {"status": "skipped"}
                    state["smoke_skipped"] += 1
                else:
                    state["current"] = identity
                    publish()

                    def cancelled():
                        state["coverage"] = _coverage(root, state["coverage"])
                        return state["coverage"] == "skip" or (
                            state["coverage"] == "representative"
                            and identity not in representatives
                        )

                    outcome = smoke(prepared[identity], cancelled=cancelled)
                    state["entries"][identity] = outcome
                    state[
                        "smoke_skipped"
                        if outcome["status"] == "skipped"
                        else "smoke_done"
                    ] += 1
                    if outcome["status"] == "failed":
                        raise RuntimeError(
                            f"preflight smoke {identity}: {outcome['error']}"
                        )
                publish()
        state.update(status="passed", finished_at=time.time(), current=None)
        publish()
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish()
        raise
