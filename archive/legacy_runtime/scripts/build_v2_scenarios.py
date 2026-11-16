"""Deterministically author V2 scenes; does not run or calibrate experiments."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCALES = {
    "small": (2, 4, 10, 300),
    "medium": (4, 5, 25, 750),
    "large": (5, 8, 50, 1500),
}
OPERATIONS = tuple("ABCDEFGH")


def capabilities(count):
    # Round-robin one-factorization of K8. Every four-machine block covers all
    # eight types once; repeated rounds differ in count by at most one.
    ring = list(OPERATIONS)
    pairs = []
    for _ in range(7):
        pairs.extend(tuple(sorted((ring[i], ring[-i - 1]))) for i in range(4))
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return [pairs[i % len(pairs)] for i in range(count)]


def scene(name, columns, rows, agvs, orders):
    baseline = yaml.safe_load(
        (ROOT / "configs/scenarios/machine-breakdown/small-320.yaml").read_text()
    )
    defaults = baseline["factory"]
    width, height = 5 * columns + 4, 3 * rows
    factory = {
        "factory_id": f"v2_{name}",
        "name": f"V2 {name}",
        "grid": {"width": width, "height": height, "blocked_cells": []},
        "operation_types": list(OPERATIONS),
        "machines": {
            "defaults": deepcopy(defaults["machines"]["defaults"]),
            "items": [],
        },
        "buffers": [],
        "inspection_stations": [],
        "scrap_bins": [],
        "chargers": [],
        "ports": [],
        "agvs": {"defaults": deepcopy(defaults["agvs"]["defaults"]), "items": []},
        "buffer_defaults": deepcopy(defaults["buffer_defaults"]),
    }

    def port(identity, owner, kind, x, y):
        factory["ports"].append(
            {
                "port_id": identity,
                "name": identity,
                "cell": {"x": x, "y": y},
                "allowed_headings": ["north", "east", "south", "west"],
                "bindings": [
                    {
                        "target": {"buffer_id": owner, "kind": "buffer"},
                        "operations": [kind],
                    }
                ],
            }
        )

    for owner, x, role, kind, px in (
        ("input", 0, "system_input", "pickup", 1),
        ("output", width - 1, "system_output", "drop_off", width - 2),
    ):
        factory["buffers"].append(
            {
                "buffer_id": owner,
                "name": owner,
                "footprint": {
                    "x": x,
                    "y": 0,
                    "width": 1,
                    "height": height,
                    "rotation": 0,
                },
                "role": role,
                "storage": {"capacity": None, "mode": "pool"},
                "machine_id": None,
            }
        )
        for index in range(columns * rows // 4):
            road_rows = [y for y in range(height) if y % 3 != 1]
            y = road_rows[
                round(index * (len(road_rows) - 1) / max(1, columns * rows // 4 - 1))
            ]
            port(f"{owner}_{index}", owner, kind, px, y)
    for index, pair in enumerate(capabilities(columns * rows)):
        x, y = 4 + 5 * (index % columns), 1 + 3 * (index // columns)
        machine = f"machine_{index:02d}"
        factory["machines"]["items"].append(
            {
                "id": machine,
                "name": machine,
                "operations": list(pair),
                "position": {"x": x, "y": y},
            }
        )
        for suffix, dx, kind in (("pre", -1, "drop_off"), ("post", 1, "pickup")):
            owner = machine + "_" + suffix
            bx = x + dx
            factory["buffers"].append(
                {
                    "buffer_id": owner,
                    "name": owner,
                    "footprint": {
                        "x": bx,
                        "y": y,
                        "width": 1,
                        "height": 1,
                        "rotation": 0,
                    },
                    "role": "machine_" + suffix,
                    "machine_id": machine,
                }
            )
            for direction, px, py in (
                ("up", bx, y - 1),
                ("down", bx, y + 1),
                ("side", bx + dx, y),
            ):
                port(owner + "_" + direction, owner, kind, px, py)
    positions = [(x, y) for y in range(0, height, 3) for x in range(2, width - 2)]
    for index in range(agvs):
        x, y = positions[index * len(positions) // agvs]
        factory["agvs"]["items"].append(
            {
                "id": f"agv_{index:02d}",
                "name": f"AGV {index}",
                "position": {"x": x, "y": y},
            }
        )
    jobs = deepcopy(baseline["jobs"])
    machines = columns * rows
    jobs["orders"] = {"total": orders, "initial": machines * 5}
    jobs["arrivals"]["mean_interval_ticks"] = 14.4 * 8 / machines
    jobs["routes"]["families"] = [
        [OPERATIONS[(i + j) % 8] for j in range(4)] for i in range(8)
    ]
    jobs["processing"]["base_ticks"] = dict(
        zip(OPERATIONS, (30.0, 40.0, 50.0, 40.0) * 2)
    )
    jobs["due_dates"] = {"mixture": [[0.2, 1.5, 2.5], [0.8, 2.0, 3.0]]}
    return {"schema": "smartsom.scenario/v1", "jobs": jobs, "factory": factory}


def main():
    from smartsom.config.scenario_authoring import ScenarioFile

    target = ROOT / "configs/scenarios/v2"
    target.mkdir(parents=True, exist_ok=True)
    for name, parameters in SCALES.items():
        data = scene(name, *parameters)
        ScenarioFile.model_validate(data)
        (target / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        print(name, "validated")


if __name__ == "__main__":
    main()
