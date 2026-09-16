"""Independent zero-defect demo ledger; does not call the simulation engine."""

import json
from collections import Counter
from pathlib import Path


def check(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text())
    scenario = manifest["inputs"]["scenario"]
    demands = {d["demand_id"]: d for d in scenario["demands"]}
    factory = scenario["factory"]
    machines = {m["machine_id"]: m for m in factory["machines"]}
    outputs = {
        b["buffer_id"] for b in factory["buffers"] if b["role"] == "system_output"
    }
    solids = {(c["x"], c["y"]) for c in factory["grid"]["blocked_cells"]}
    for group in (
        "machines",
        "buffers",
        "inspection_stations",
        "scrap_bins",
        "chargers",
    ):
        for resource in factory[group]:
            f = resource["footprint"]
            solids.update(
                (x, y)
                for x in range(f["x"], f["x"] + f["width"])
                for y in range(f["y"], f["y"] + f["height"])
            )
    capacities = {}
    for b in factory["buffers"]:
        storage = b["storage"]
        if "slots" in storage:
            capacities[b["buffer_id"]] = {
                s["slot_id"]: s["capacity"] for s in storage["slots"]
            }
        else:
            capacities[b["buffer_id"]] = {"pool": storage["capacity"]}
    for q in factory["inspection_stations"]:
        capacities[q["inspection_station_id"]] = {
            s["slot_id"]: s["capacity"] for s in q["slots"]
        }
    starts, steps, deliveries, samples = {}, Counter(), set(), []
    transport = {}
    transport_samples = []
    previous = manifest["initial_state"]
    last = previous
    for line in (directory / "trace.jsonl").open():
        row = json.loads(line)
        state = row["state"]
        cells = [tuple(v["cell"]) for v in state["agvs"].values()]
        assert len(cells) == len(set(cells)), "vehicle overlap"
        assert all(
            c not in solids
            and 0 <= c[0] < factory["grid"]["width"]
            and 0 <= c[1] < factory["grid"]["height"]
            for c in cells
        ), "obstacle/bounds"
        assert set(state["released"]) == {
            d for d, spec in demands.items() if spec["release_at"] <= state["tick"]
        }, "release boundary"
        for aid, vehicle in state["agvs"].items():
            old = previous["agvs"][aid]["cell"]
            assert sum(abs(a - b) for a, b in zip(old, vehicle["cell"])) <= 1, (
                "teleport"
            )
            for bid in state["agvs"]:
                if aid != bid and old != vehicle["cell"]:
                    assert not (
                        old == state["agvs"][bid]["cell"]
                        and vehicle["cell"] == previous["agvs"][bid]["cell"]
                    ), "swap"
        owners = Counter()
        for owner, slots in state["storage"].items():
            for slot, jobs in slots.items():
                cap = capacities[owner][slot]
                assert cap is None or len(jobs) <= cap, "capacity exceeded"
                for job in jobs:
                    owners[job] += 1
                    assert state["jobs"][job]["location"] == owner
        for owner, resource in {**state["agvs"], **state["machines"]}.items():
            if resource["job"]:
                owners[resource["job"]] += 1
                assert state["jobs"][resource["job"]]["location"] == owner
        for job in state["queue"]:
            owners[job] += 1
        assert set(owners) == set(state["jobs"]) and all(
            v == 1 for v in owners.values()
        ), "ownership"
        assert Counter(j["demand"] for j in state["jobs"].values()) == Counter(
            state["released"]
        ), "demand conservation"
        for e in row["events"]:
            kind = e["kind"]
            if kind == "pickup":
                assert e["agv"] not in transport
                transport[e["agv"]] = {
                    "job": e["job"],
                    "pickup": e["tick"],
                    "moves": 0,
                    "source": e["owner"],
                    "start": previous["agvs"][e["agv"]]["cell"],
                }
            if kind == "move" and e["agv"] in transport:
                transport[e["agv"]]["moves"] += 1
            if kind == "drop":
                trip = transport.pop(e["agv"])
                assert trip["job"] == e["job"]
                assert e["tick"] - trip["pickup"] >= trip["moves"] + 1, "transfer time"
                if e["job"].split("/")[0] in {
                    "order_01",
                    "order_02",
                    "order_03",
                    "order_04",
                }:
                    transport_samples.append(
                        {
                            **trip,
                            "drop": e["tick"],
                            "destination": e["owner"],
                            "end": state["agvs"][e["agv"]]["cell"],
                        }
                    )
            if kind == "demand_released":
                assert e["tick"] == demands[e["demand"]]["release_at"]
            if kind == "processing_started":
                assert e["machine"] not in starts, "machine double start"
                d = demands[state["jobs"][e["job"]]["demand"]]
                step = d["steps"][steps[e["job"]]]
                assert (
                    step["operation_type"] in machines[e["machine"]]["operation_types"]
                )
                assert e["actual_ticks"] == step["nominal_ticks"], "duration mismatch"
                starts[e["machine"]] = (e, step)
            if kind == "processing_completed":
                start, step = starts.pop(e["machine"])
                assert (
                    start["job"] == e["job"] and step["operation_id"] == e["operation"]
                )
                active = sum(
                    not any(
                        o["machine_id"] == e["machine"] and o["start"] <= t < o["end"]
                        for o in scenario["outages"]
                    )
                    for t in range(start["tick"], e["tick"])
                )
                assert active == step["nominal_ticks"], "premature/late completion"
                assert e["elapsed"] == active and not e["defect"]
                steps[e["job"]] += 1
                if state["jobs"][e["job"]]["demand"] in {
                    "order_01",
                    "order_02",
                    "order_03",
                    "order_04",
                }:
                    samples.append(
                        dict(
                            job=e["job"],
                            machine=e["machine"],
                            start=start["tick"],
                            end=e["tick"],
                            active_ticks=active,
                        )
                    )
            if kind == "drop" and e["owner"] in outputs:
                demand = state["jobs"][e["job"]]["demand"]
                assert steps[e["job"]] == len(demands[demand]["steps"]), (
                    "skipped operation"
                )
                assert demand not in deliveries, "duplicate delivery"
                assert state["jobs"][e["job"]]["quality"] == "PASS"
                deliveries.add(demand)
        assert set(state["completed"]) == deliveries, "delivery counter disagreement"
        previous = last = state
    assert deliveries == set(demands), "incomplete delivery"
    return dict(
        passed=True,
        ticks=last["tick"],
        qualified=len(deliveries),
        operations=sum(steps.values()),
        samples=samples,
        transport_samples=transport_samples,
    )


if __name__ == "__main__":
    import sys

    print(json.dumps(check(sys.argv[1]), indent=2))
