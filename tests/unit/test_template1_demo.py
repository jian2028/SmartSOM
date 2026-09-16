"""Acceptance gates for the coordinated Template 1 example."""

from dataclasses import replace
from pathlib import Path

from smartsom import api
from smartsom.algorithms.coordinated import CoordinatedProductionPolicy
from smartsom.domain.production import Outage
from smartsom.engine.production import ProductionSimulator

ROOT = Path(__file__).resolve().parents[2]


def scenario(name="warmup"):
    return api.prepare(
        api.load_config(ROOT / f"configs/runs/template1_{name}.yaml"), training=False
    ).resolved.scenario


def step(sim, policy):
    view = sim.decision(policy.rank(sim.decision()))
    return sim.step(policy.act(view))


def test_template_layout_preserved():
    assert (ROOT / "configs/factories/template1_demo.yaml").read_bytes() == (
        ROOT / "src/smartsom/studio/templates/template_001.yaml"
    ).read_bytes()


def test_nominal_route_public_but_unreleased_jobs_absent():
    sim = ProductionSimulator(scenario("disturbed"))
    view = sim.decision()
    assert len(view["jobs"]) == 12
    assert all(
        j["demand"] not in {"order_13", "order_14", "order_15", "order_16"}
        for j in view["jobs"].values()
    )
    assert [
        s["nominal_ticks"] for s in next(iter(view["jobs"].values()))["remaining_steps"]
    ] == [4, 6, 8]


def test_template_processing_outage_hand_check():
    s = replace(scenario(), outages=(Outage("machine_001", 10, 13),))
    sim, policy = ProductionSimulator(s), CoordinatedProductionPolicy(s.factory)
    events = []
    while not sim.done:
        row = step(sim, policy)
        events.extend(row["events"])
    starts = [e for e in events if e["kind"] == "processing_started"]
    ends = [e for e in events if e["kind"] == "processing_completed"]
    assert starts[0]["tick"] == 9
    assert ends[0]["tick"] == 16  # 4 productive ticks + three paused ticks.
    assert ends[0]["elapsed"] == 4
    assert len(sim.completed) == 1


def test_routes_respect_vertex_and_reverse_edge_reservations():
    policy = CoordinatedProductionPolicy(scenario().factory)
    route = policy.route((3, 4), (3, 5), {((3, 5), 1)}, set())
    assert route and route[0] != "DOWN"
    route = policy.route((3, 4), (3, 5), set(), {((3, 5), (3, 4), 1)})
    assert route and route[0] != "DOWN"


def test_independent_ledger_rejects_corrupted_counters_and_timing(tmp_path):
    import importlib.util
    import json

    import pytest

    from smartsom.config.production import AlgorithmConfig
    from smartsom.experiments.production import execute

    spec = importlib.util.spec_from_file_location(
        "demo_audit", ROOT / "scripts/validation/template1_audit.py"
    )
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    directory = execute(
        scenario(),
        AlgorithmConfig(provider="builtin.coordinated"),
        output_root=tmp_path,
        verbose=False,
    )
    assert audit_module.check(directory)["passed"]
    trace = directory / "trace.jsonl"
    original = trace.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[-1]["state"]["completed"] = []
    trace.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError, match="counter"):
        audit_module.check(directory)
    rows = [json.loads(line) for line in original.splitlines()]
    event = next(
        e for row in rows for e in row["events"] if e["kind"] == "processing_completed"
    )
    event["tick"] -= 1
    trace.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError, match="completion"):
        audit_module.check(directory)
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["state"]["agvs"]["agv_001"]["cell"] = [11, 11]
    trace.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError, match="obstacle|teleport"):
        audit_module.check(directory)


def test_training_recipes_resolve_intended_network_and_full_episode_limits():
    for name in ("warmup", "static", "disturbed"):
        config = api.load_config(ROOT / f"configs/runs/template1_train_{name}.yaml")
        prepared = api.prepare(config, training=False)
        assert prepared.resolved.algorithm.hidden_sizes == (128, 128)
        assert prepared.resolved.algorithm.max_jobs == 20
        assert config.training.max_decisions == 100000
        assert config.training.max_ticks == 2000


def test_transport_memory_resets_with_new_episode():
    s = scenario()
    policy = CoordinatedProductionPolicy(s.factory)
    sim = ProductionSimulator(s)
    for _ in range(2):
        step(sim, policy)
    assert policy.tasks
    policy.rank(ProductionSimulator(s).decision())
    assert policy.tasks == {}
