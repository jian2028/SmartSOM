"""Current quality/replacement runs plus a separate seeded draw-distribution check.

The distribution check measures Bernoulli draws and permanent-defect composition;
it is not a completed simulation or a learned-policy performance claim.
"""

import json
import math
from dataclasses import replace

from validation.grid_cases import ROOT, main

from smartsom.api import load_config, prepare
from smartsom.engine.production import ProductionSimulator


def statistics():
    spec = json.loads((ROOT / "data/reference/quality/reference.json").read_text())[
        "statistics"
    ]
    case = prepare(
        load_config(ROOT / "configs/runs/quality_m0.yaml"), training=False
    ).resolved.scenario
    rates = {
        mode.quality_mode_id: float(mode.error_rate)
        for mode in case.factory.machines[0].quality_modes
    }
    counts = [0] * len(spec["routes"])
    first, size = spec["root_seed_start"], spec["samples_per_group"]
    for seed in range(first, first + size):
        sim = ProductionSimulator(replace(case, seed=seed, quality_samples=()))
        draws = [
            sim._draw("quality", "statistics/attempt/1", f"op{i}") for i in range(5)
        ]
        for group, route in enumerate(spec["routes"]):
            counts[group] += all(
                draw >= rates[mode] for draw, mode in zip(draws, route, strict=True)
            )
    rows = []
    for route, count in zip(spec["routes"], counts, strict=True):
        expected = math.prod(1 - rates[mode] for mode in route)
        error = abs(count / size - expected)
        rows.append(
            {
                "route": route,
                "passed_draw_sequences": count,
                "samples": size,
                "expected": expected,
                "observed": count / size,
                "absolute_error": error,
                "tolerance": spec["absolute_tolerance"],
                "passed": error <= spec["absolute_tolerance"],
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(
        main(
            (
                "quality_m0",
                "quality_m1",
                "quality_m2",
                "quality_generated",
                "quality_hidden",
                "quality_machine",
                "quality_combined",
            ),
            description=__doc__,
            statistics=statistics,
        )
    )
