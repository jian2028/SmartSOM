"""Bundled examples run only on explicit grid layouts and semantic commands."""

import json
from pathlib import Path

import pytest

from smartsom.api import load_config, prepare
from smartsom.config.codec import primitive
from smartsom.config.production import scenario_from_snapshot
from smartsom.experiments.runner import run_one
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[2]
RUNS = sorted((ROOT / "configs/runs").glob("*.yaml"))


@pytest.mark.parametrize("source", RUNS, ids=lambda p: p.stem)
def test_every_bundled_run_uses_grid_or_explicitly_rejects_cp_sat(source):
    config = load_config(source)
    if source.stem.endswith("_cp"):
        with pytest.raises(ValueError, match="CP-SAT has no grid production adapter"):
            prepare(config, training=False)
        return
    recipe = prepare(config, training=False).resolved
    assert scenario_from_snapshot(primitive(recipe.scenario)) == recipe.scenario
    assert recipe.scenario.factory.ports
    assert all(m.operation_types for m in recipe.scenario.factory.machines)


@pytest.mark.parametrize(
    "name,ticks,completed",
    [
        ("fjsp_fast", 32, 2),
        ("fjsp_slow", 33, 2),
        ("crossing", 32, 2),
        ("buffers_direct_zero", 39, 3),
        ("buffers_vehicle_zero", 26, 2),
        ("buffers_post_one", 34, 3),
        ("holding_hand", 22, 1),
    ],
)
def test_scripted_examples_complete_and_audit(name, ticks, completed, tmp_path):
    config = load_config(ROOT / "configs/runs" / f"{name}.yaml")
    config.output.root = str(tmp_path)
    config.logging.verbose = False
    result = run_one(prepare(config, training=False))
    assert result.simulation_result.status == "completed"
    assert result.simulation_result.makespan == ticks
    assert result.simulation_result.qualified_demands == completed
    assert audit(result.run_dir)["status"] == "passed"
    records = [
        json.loads(line)
        for line in (result.run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert not any(row["rejections"] for row in records)
    if name == "buffers_post_one":
        assert any(
            row["state"]["machines"]["M1"]["status"] == "BLOCKED" for row in records
        )
        assert any(row["state"]["storage"]["M1_post"]["slot_1"] for row in records)
    if name == "holding_hand":
        assert any(row["state"]["storage"]["b-hold"]["slot_1"] for row in records)


def test_migrated_fixed_samples_preserve_the_explicit_values():
    recipe = prepare(
        load_config(ROOT / "configs/runs/quality_combined.yaml"), training=False
    ).resolved
    case = recipe.scenario
    assert {
        (s.operation_id, s.machine_id, s.actual_ticks) for s in case.processing_samples
    } == {
        ("A", "M1", 3),
        ("B", "M2", 4),
        ("K", "M2", 3),
    }
    assert {(o.machine_id, o.start, o.end) for o in case.outages} == {
        ("M1", 6, 9),
        ("M2", 8, 10),
    }
    assert {(d.demand_id, d.release_at, d.reveal_at) for d in case.demands} == {
        ("J", 0, 0),
        ("K", 4, 2),
    }
