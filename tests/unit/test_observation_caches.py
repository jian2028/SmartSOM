"""Observation and public-fact caches must reproduce the uncached values exactly."""

import hashlib
import json
import random
from collections.abc import Mapping

import pytest

from smartsom.config.base_diagnostic import BaseWorkload, generate_scenario
from smartsom.config.base_layout import base_factory
from smartsom.controllers.e1a.observation import create_observation
from smartsom.controllers.morris import observation as morris
from smartsom.controllers.morris.observation import global_inputs, one_hot
from smartsom.domain.public_facts import Facts, freeze, plain
from smartsom.engine.central_control import CentralDecision
from smartsom.engine.central_environment import CentralEnvironment
from smartsom.engine.public_facts import public_facts
from tests.helpers.algorithm import algorithm_options

T, N = 200.0, 20.0


def reference_global_inputs(facts, t, n):
    """The uncached implementation the cached one replaces."""
    machines = sorted(facts["machines"], key=lambda r: r["id"])
    vehicles = sorted(facts["vehicles"], key=lambda r: r["id"])
    buffers = sorted(facts["buffers"], key=lambda r: r["id"])
    owners = tuple(sorted(r["id"] for r in (*machines, *vehicles, *buffers)))
    topology = facts["topology"]
    types = topology["operation_types"]
    width, height = topology["width"], topology["height"]
    jobs = {r["id"]: r for r in facts["jobs"]}

    def summary(ids):
        rows = [jobs[j] for j in ids if j in jobs]
        if not rows:
            return (0.0,) * 8
        slack = [(r["due_at"] - facts["tick"]) / t for r in rows]
        return (
            len(rows) / n,
            min(slack),
            sum(slack) / len(rows),
            sum(x < 0 for x in slack) / n,
            sum(r["rush"] for r in rows) / n,
            sum(r["step"] for r in rows) / n,
            sum(len(r["remaining_operations"]) for r in rows) / n,
            sum(r["public_risk"] for r in rows) / len(rows),
        )

    state = [facts["tick"] / t, len(facts["completed"]) / n, *summary(jobs)]
    for row in machines:
        state.extend(
            (
                float(row["down"]),
                float(row["status"] == "PROCESSING"),
                float(row["status"] == "BLOCKED"),
                float(row["job"] is not None),
                row["elapsed"] / t,
                row["nominal"] / t,
                *one_hot(row["mode"], ("slow", "normal", "fast")),
                *(float(k in row["capabilities"]) for k in types),
                *summary((row["job"],)),
            )
        )
    for row in vehicles:
        target = row["target"]
        state.extend(
            (
                row["cell"][0] / width,
                row["cell"][1] / height,
                float(row["job"] is not None),
                float(row["service"] is not None),
                (facts["tick"] - row["arrived_at"]) / t,
                *one_hot(target["owner"] if target else None, owners),
                *summary((row["job"],)),
            )
        )
    for row in buffers:
        capacities = tuple(row["capacity"].values())
        inventory = [j for values in row["inventory"].values() for j in values]
        state.extend(
            (
                sum(c or 0 for c in capacities) / n,
                float(any(c is None for c in capacities)),
                row["quota"] / n,
                len(row["commitments"]) / n,
                len(row["ready"]) / n,
                *summary(inventory),
            )
        )
    solids = set(topology["solids"])
    state.extend(float((x, y) in solids) for y in range(height) for x in range(width))
    for owner in owners:
        cells = topology["ports"].get(owner, ())
        state.extend(
            (
                len(cells) / n,
                sum(c[0] / width for c in cells) / max(1, len(cells)),
                sum(c[1] / height for c in cells) / max(1, len(cells)),
            )
        )
    layout = (
        owners,
        types,
        plain(topology),
        tuple((r["id"], r["capabilities"]) for r in machines),
    )
    signature = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return tuple(state), owners, signature


def reference_association(facts, candidate):
    machines = {r["id"]: r for r in facts["machines"]}
    return candidate.machine or next(
        (
            m["id"]
            for m in machines.values()
            if candidate.destination in (m["pre"], m["post"])
        ),
        None,
    )


def reference_snapshot(core):
    """Fresh row caches, then the former freeze-twice topology construction."""
    for name in ("_public_job_rows", "_public_operation_rows"):
        if hasattr(core.protocol, name):
            delattr(core.protocol, name)
    plain_facts = public_facts(core)
    return Facts((*plain_facts.entries, ("topology", core.protocol.public_topology)))


def environment(seed, orders=8):
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=orders,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    options = algorithm_options(method="rule", control="centralized")
    scenario = generate_scenario(base_factory("small"), workload, seed, "high", 1000)
    return CentralEnvironment(scenario, options)


def same(a, b):
    """Type-strict, bit-exact structural equality (1, 1.0 and True differ)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex() or (a != a and b != b)
    if isinstance(a, Mapping):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(map(same, a, b))
    return a == b


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    monkeypatch.setattr(morris, "_STATIC", {})
    monkeypatch.setattr(morris, "_JOBS", {})


def trajectory(env, rng, limit=400):
    """Yield (snapshot, request) at every central decision of a random walk."""
    actor = create_observation(time_scale=T, count_scale=N)
    for _ in range(limit):
        candidates = env.info()["candidates"]
        if env.core.done or not candidates:
            return
        facts = public_facts(env.core, topology=True)
        yield facts, CentralDecision(env.core.tick, candidates, None), actor
        env.step(rng.choice(candidates).identity)


CASES = [(3, 8), (17, 8), (11, 30)]


@pytest.mark.parametrize("seed, orders", CASES)
def test_cached_observations_match_uncached_reference_on_a_trajectory(seed, orders):
    env, rng, checked = environment(seed, orders), random.Random(seed), 0
    for facts, request, actor in trajectory(env, rng):
        expected = reference_global_inputs(facts, T, N)
        assert same(global_inputs(facts, T, N), expected)
        # Warm caches must return the same values on a repeated read.
        assert same(global_inputs(facts, T, N), expected)
        observed = actor(facts, request)
        assert same(observed["global"], expected[0])
        assert observed["layout"] == expected[2]
        assert [a[0] for a in observed["associations"]] == [
            reference_association(facts, c) for c in request.candidates
        ]
        checked += 1
    assert checked > 50
    assert env.core.tick > 0


@pytest.mark.parametrize("seed, orders", CASES)
def test_incremental_public_snapshot_matches_fresh_construction(seed, orders):
    env, rng, checked = environment(seed, orders), random.Random(seed), 0
    for facts, request, _ in trajectory(env, rng):
        # Unchanged job rows are shared objects between snapshots.
        again = public_facts(env.core, topology=True)
        assert all(a is b for a, b in zip(again["jobs"], facts["jobs"], strict=True))
        expected = reference_snapshot(env.core)
        assert same(facts, expected)
        assert same(public_facts(env.core, topology=True), expected)
        checked += 1
    assert checked > 50


@pytest.mark.parametrize(
    "field, value",
    [
        ("location", "elsewhere"),
        ("slot", 99),
        ("since", 12345),
        ("quality", "FAIL"),
        ("risk", 0.75),
        ("step", None),
    ],
)
def test_every_mutable_job_field_invalidates_the_cached_row(field, value):
    core = environment(5).core
    job = next(iter(core.jobs))
    first = public_facts(core)["jobs"][0]
    assert first["id"] == job
    assert public_facts(core)["jobs"][0] is first
    core.jobs[job][field] = core.jobs[job]["step"] + 1 if field == "step" else value
    changed = public_facts(core)["jobs"][0]
    assert changed is not first
    assert same(changed, reference_snapshot(core)["jobs"][0])


def test_job_row_reflects_progress_and_demand_replacement():
    core = environment(5).core
    job = next(iter(core.jobs))
    first = public_facts(core)["jobs"][0]
    core.jobs[job]["step"] += 1
    stepped = public_facts(core)["jobs"][0]
    assert stepped["step"] == first["step"] + 1
    assert (
        len(stepped["remaining_operations"]) == len(first["remaining_operations"]) - 1
    )

    from dataclasses import replace

    demand = core.jobs[job]["demand"]
    core.demands[demand] = replace(core.demands[demand], due_at=10**6, rush=True)
    replaced = public_facts(core)["jobs"][0]
    assert (replaced["due_at"], replaced["rush"]) == (10**6, True)
    assert same(replaced, reference_snapshot(core)["jobs"][0])


def test_static_layout_cache_is_per_topology_object_and_layout():
    a, b = environment(3), environment(3)
    fa = public_facts(a.core, topology=True)
    fb = public_facts(b.core, topology=True)
    assert fa["topology"] is not fb["topology"]
    assert same(global_inputs(fa, T, N), reference_global_inputs(fa, T, N))
    assert same(global_inputs(fb, T, N), reference_global_inputs(fb, T, N))
    keys = {key[0] for key in morris._STATIC}
    assert {id(fa["topology"]), id(fb["topology"])} <= keys
    # A different count scale is a different layout encoding.
    assert same(global_inputs(fa, T, 7.0), reference_global_inputs(fa, T, 7.0))
    # Same topology object, different machines, capabilities or vehicles.
    machines = fa["machines"]
    for name, variant in (
        ("machines", machines[1:]),
        (
            "machines",
            (Facts({**machines[0], "capabilities": ("Z",)}.items()), *machines[1:]),
        ),
        ("vehicles", fa["vehicles"][1:]),
    ):
        changed = Facts({**fa, name: variant}.items())
        assert changed["topology"] is fa["topology"]
        assert same(
            global_inputs(changed, T, N), reference_global_inputs(changed, T, N)
        )


def test_job_cache_holds_only_the_latest_snapshot():
    env, rng = environment(11, 30), random.Random(11)
    for facts, request, _ in trajectory(env, rng, limit=200):
        global_inputs(facts, T, N)
        assert set(morris._JOBS) == {id(r) for r in facts["jobs"]}
    for n in range(1, 40):
        global_inputs(facts, T, float(n))
    assert len(morris._STATIC) <= 8


def test_mutable_inputs_are_computed_but_never_cached():
    facts = public_facts(environment(3).core, topology=True)
    expected = reference_global_inputs(facts, T, N)
    mutable = {**facts, "topology": dict(facts["topology"])}
    mutable["jobs"] = tuple(dict(r) for r in facts["jobs"])
    assert same(global_inputs(mutable, T, N), expected)
    assert morris._STATIC == {} and morris._JOBS == {}
    # In-place edits of the mutable copy are reflected on the next call.
    mutable["jobs"][0]["public_risk"] = 0.5
    assert same(global_inputs(mutable, T, N), reference_global_inputs(mutable, T, N))


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, 2.5, "a", None, True], (1, 2.5, "a", None, True)),
        ((1.0, 2.0), (1.0, 2.0)),
        ([1, [2, 3]], (1, (2, 3))),
        ((1, {"a": [2]}), (1, Facts((("a", (2,)),)))),
        ([], ()),
    ],
)
def test_freeze_fast_path_matches_recursive_freeze(value, expected):
    frozen = freeze(value)
    assert type(frozen) is tuple
    assert same(frozen, expected)
    if isinstance(value, tuple) and all(type(v) is float for v in value):
        assert frozen is value


def test_freeze_reuses_frozen_rows_and_rejects_private_objects():
    row = Facts((("id", "j1"),))
    rows = (row, row)
    assert freeze(rows) is rows
    assert freeze([row])[0] is row
    with pytest.raises(TypeError):
        freeze((1, object()))
    with pytest.raises(TypeError):
        freeze([1, [object()]])


def test_cache_entries_for_a_reused_object_id_are_not_trusted():
    facts = public_facts(environment(3).core, topology=True)
    topology = facts["topology"]
    expected = reference_global_inputs(facts, T, N)
    machines = sorted(facts["machines"], key=lambda r: r["id"])
    capabilities = tuple((r["id"], r["capabilities"]) for r in machines)
    key = id(topology), expected[1], capabilities, type(N), N
    # Simulate freed objects whose ids now belong to the live topology and row.
    morris._STATIC[key] = object(), ((), "stale")
    morris._JOBS[id(facts["jobs"][0])] = object(), (0, 0, 0, 0, 0.0)
    assert same(global_inputs(facts, T, N), expected)


def test_destination_association_keeps_first_machine_in_public_order(monkeypatch):
    import smartsom.controllers.e1a.observation as e1a
    from smartsom.engine.central_control import task

    monkeypatch.setattr(e1a, "global_inputs", lambda facts, t, n: ((), (), "layout"))
    machine = dict(status="IDLE", nominal=1.0)
    facts = {
        "tick": 0,
        "topology": {"width": 4, "height": 4},
        "jobs": (),
        "vehicles": (),
        "buffers": (),
        # M2 is listed first and shares B with M1; ADVANCE has no destination.
        "machines": (
            dict(machine, id="M2", pre="B", post=None),
            dict(machine, id="M1", pre="A", post="B"),
        ),
    }
    candidates = (
        task("transport", job="j", vehicle="v", destination="B"),
        task("transport", job="j", vehicle="v", destination="A"),
        task("advance"),
    )
    observed = create_observation()(facts, CentralDecision(0, candidates, None))
    expected = [reference_association(facts, c) for c in candidates]
    assert [a[0] for a in observed["associations"]] == expected == ["M2", "M1", "M2"]
