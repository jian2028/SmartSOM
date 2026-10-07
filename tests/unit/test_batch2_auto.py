"""Focused Batch 02 reward and diagnostic-physics contracts."""

from types import SimpleNamespace

from smartsom.learning.extensions import dispatcher_pickup_opportunity


def test_missed_pickup_is_one_boundary_even_with_multiple_eligible_agvs():
    before = {
        "agvs": {
            "a": {"job": None, "reservation": None},
            "b": {"job": None, "reservation": None},
        }
    }
    actions = [
        {
            "role": "dispatcher",
            "owner": owner,
            "candidate": "NO_REQUEST",
            "candidates": [{"identity": "TARGET:input", "legal": True}],
        }
        for owner in ("a", "b")
    ]
    assert dispatcher_pickup_opportunity(before, actions) == (2, True)
    actions[0]["candidate"] = "TARGET:input"
    assert dispatcher_pickup_opportunity(before, actions) == (2, False)
    actions[0]["candidates"][0]["legal"] = False
    assert dispatcher_pickup_opportunity(before, actions) == (1, True)


def test_dispatch_candidates_follow_current_cargo_and_operation():
    from smartsom import api
    from smartsom.algorithms.production_composition import BoundaryCoordinator
    from smartsom.algorithms.production_rules import RulePolicy
    from smartsom.engine.production import ProductionSimulator

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    scenario = api.prepare(
        api.load_config(root / "configs/test/runs/template1_warmup.yaml"),
        training=False,
    ).resolved.scenario
    sim = ProductionSimulator(scenario, contract="v3")
    empty = next(iter(sim.agvs))
    sim.protocol.begin()
    initial = sim.protocol.dispatch_candidates(empty)
    assert initial and all(c.identity.startswith("TARGET:") for c in initial)
    assert {c.action.owner for c in initial} == set(sim.protocol.sources)
    sim.protocol.abort()
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "shortest_path",
    }
    driver = BoundaryCoordinator(
        sim,
        {role: RulePolicy(role, name) for role, name in names.items()},
        {role: {"default": role} for role in names},
        "global_optimal",
    )
    for _ in range(25):
        driver.tick()
        loaded = [car for car, state in sim.agvs.items() if state["job"]]
        if loaded:
            break
    assert loaded
    car = loaded[0]
    sim.protocol.begin()
    targets = sim.protocol.dispatch_candidates(car)
    assert targets and all(
        candidate.identity.startswith("TARGET:") for candidate in targets
    )
    assert {candidate.action.owner for candidate in targets} <= set(
        sim.protocol.destination_owners(sim.agvs[car]["job"])
    )
    sim.protocol.abort()


def test_dispatcher_penalty_applies_once_per_boundary():
    from smartsom.learning.extensions import _DispatcherMissedPickup

    before = {
        "agvs": {name: {"job": None, "reservation": None} for name in ("a", "b", "c")}
    }
    actions = [
        {
            "role": "dispatcher",
            "owner": name,
            "candidate": "NO_REQUEST",
            "candidates": [{"identity": "TARGET:input", "legal": True}],
        }
        for name in before["agvs"]
    ]
    transition = SimpleNamespace(
        role="dispatcher_policy", before=before, actions=actions
    )
    reward = _DispatcherMissedPickup({"penalty": 0.05})
    assert reward.transform(transition, 1.0) == 0.95
    actions[0]["candidate"] = "TARGET:input"
    assert reward.transform(transition, 1.0) == 1.0
