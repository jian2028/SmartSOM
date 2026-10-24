"""Copy equivalence for detached, scalar-heavy boundary records."""

import copy
from dataclasses import dataclass
from decimal import Decimal

import pytest

from smartsom._copy import detached_copy
from smartsom.config.codec import primitive


def test_nested_records_remain_detached_and_preserve_shared_rows():
    row = {"scores": [1.0, 2.0], "state": {"tick": 3, "job": None}}
    original = {"observation": row, "model_input": row, "prefix": (row,)}
    result = detached_copy(original)
    assert result == copy.deepcopy(original)
    assert result["observation"] is result["model_input"] is result["prefix"][0]
    result["observation"]["scores"].append(4)
    result["model_input"]["state"]["tick"] = 99
    assert row == {"scores": [1.0, 2.0], "state": {"tick": 3, "job": None}}


@pytest.mark.parametrize("kind", [list, dict, tuple])
def test_recursive_graphs_preserve_cycles(kind):
    if kind is list:
        original = []
        original.append(original)
        result = detached_copy(original)
        assert result is result[0]
    elif kind is dict:
        original = {}
        original["self"] = original
        result = detached_copy(original)
        assert result is result["self"]
    else:
        child = []
        original = (child,)
        child.append(original)
        result = detached_copy(original)
        assert result is result[0][0]
    assert result is not original


def test_unknown_copy_hooks_share_the_container_memo():
    class Extension:
        def __init__(self, row):
            self.row = row

        def __deepcopy__(self, memo):
            result = Extension(copy.deepcopy(self.row, memo))
            memo[id(self)] = result
            return result

    row = [3.5]
    result = detached_copy([row, Extension(row)])
    assert result[0] is result[1].row
    assert result[0] is not row


def test_builtin_subclasses_retain_copy_hooks():
    class Scores(list):
        def __deepcopy__(self, memo):
            return {"custom": list(self)}

    assert detached_copy({"scores": Scores([1])}) == {"scores": {"custom": [1]}}


@pytest.mark.parametrize("kind", [float, list, dict, tuple])
def test_changed_copy_dispatch_declines_fast_path(monkeypatch, kind):
    monkeypatch.setitem(copy._deepcopy_dispatch, kind, lambda value, memo: "custom")
    value = {float: 1.0, list: [1], dict: {"a": 1}, tuple: (1,)}[kind]
    assert detached_copy(value) == copy.deepcopy(value) == "custom"


def test_changed_deepcopy_function_is_honored(monkeypatch):
    monkeypatch.setattr(copy, "deepcopy", lambda value: {"custom": value})
    assert detached_copy([1]) == {"custom": [1]}


def test_primitive_retains_dataclass_decimal_and_subclass_semantics():
    @dataclass
    class Row:
        score: Decimal
        child: list

    child = [None, True, 2, 3.5, "value"]
    result = primitive(Row(Decimal("1.25"), child))
    assert result == {"score": "1.25", "child": child}
    assert result["child"] is not child


def test_full_boundary_records_match_stdlib_copy_and_remain_isolated(monkeypatch):
    from test_composable_physics import coordinator, example

    import smartsom._copy as copying
    import smartsom.algorithms.production_composition as composition
    import smartsom.engine.production as production
    import smartsom.engine.production_protocol as protocol
    from smartsom.engine.production import ProductionSimulator

    scenario = example()
    fast = coordinator(ProductionSimulator(scenario, contract="v3"))
    baseline = coordinator(ProductionSimulator(scenario, contract="v3"))
    saw_decisions = False
    for _ in range(100):
        if fast.sim.done:
            break
        fast_record = fast.tick()
        with monkeypatch.context() as patch:
            patch.setattr(copying, "detached_copy", copy.deepcopy)
            patch.setattr(composition, "detached_copy", copy.deepcopy)
            patch.setattr(protocol, "detached_copy", copy.deepcopy)
            patch.setattr(production, "detached_copy", copy.deepcopy)
            patch.setattr(protocol, "step_descriptor", protocol.asdict)
            baseline_record = baseline.tick()
        assert fast_record == baseline_record
        if fast_record["decisions"]:
            saw_decisions = True
            saved = copy.deepcopy(fast_record)
            fast.records[0]["observation"]["tick"] = -1
            assert fast_record == saved
    assert saw_decisions
    assert fast.sim.snapshot() == baseline.sim.snapshot()


@pytest.mark.parametrize(
    "container", ["direct", "list", "dict_key", "dict_value", "tuple"]
)
@pytest.mark.parametrize("replacement", ["replacement", None, ["replacement"]])
def test_extension_memo_replacements_survive_later_scalar_containers(
    container, replacement
):
    value = float("3.25")

    class Extension:
        def __deepcopy__(self, memo):
            memo[id(value)] = replacement
            return "copied"

    payload = {
        "direct": value,
        "list": [value],
        "dict_key": {value: "item"},
        "dict_value": {"item": value},
        "tuple": (value,),
    }[container]
    graph = [Extension(), payload]
    if container == "dict_key" and isinstance(replacement, list):
        with pytest.raises(TypeError, match="unhashable"):
            copy.deepcopy(graph)
        with pytest.raises(TypeError, match="unhashable"):
            detached_copy(graph)
    else:
        result = detached_copy(graph)
        assert result == copy.deepcopy(graph)
        if container != "dict_key":
            child = (
                result[1]
                if container == "direct"
                else result[1]["item" if container == "dict_value" else 0]
            )
            assert child is replacement


def test_extension_copy_registration_changes_apply_to_remaining_graph(monkeypatch):
    value = float("3.25")
    results = []
    for copier in (copy.deepcopy, detached_copy):
        with monkeypatch.context() as patch:

            class Extension:
                def __deepcopy__(self, memo):
                    patch.setitem(
                        copy._deepcopy_dispatch, float, lambda value, memo: "custom"
                    )
                    return "copied"

            results.append(copier([Extension(), [value], {"value": value}]))
    assert results[0] == results[1] == ["copied", ["custom"], {"value": "custom"}]


@pytest.mark.parametrize("container", [list, dict, tuple])
def test_extension_cannot_leave_completed_container_memo_overwritten(container):
    shared = None

    class Extension:
        def __deepcopy__(self, memo):
            memo[id(shared)] = "replacement"
            return "copied"

    shared = {"value": Extension()} if container is dict else container([Extension()])
    graph = [shared, shared]
    result = detached_copy(graph)
    expected = copy.deepcopy(graph)
    assert result == expected
    assert result[0] is result[1]


def test_public_view_fresh_same_tick_stage_mutations_and_private_projection(
    monkeypatch,
):
    from test_production_runtime import small_scenario

    import smartsom.engine.production as production
    from smartsom.engine.production import ProductionSimulator

    sim = ProductionSimulator(small_scenario(), contract="v3")
    protocol = sim.protocol
    protocol.begin()
    first = protocol.public_view()
    before = copy.deepcopy(first)
    job = next(iter(sim.jobs))
    sim.jobs[job]["risk"] = 0.125
    sim.jobs[job]["defective"] = True
    sim.agvs["agv"]["feedback"] = {"state": ["fresh"]}
    sim.agvs["agv"]["target"] = {"owner": "input", "private": ["target"]}
    sim.rankings["input"] = [job]
    sim.storage["input"][None] = []
    second = protocol.public_view()
    with monkeypatch.context() as patch:
        patch.setattr(production, "detached_copy", copy.deepcopy)
        expected = protocol.public_view()
    assert first == before
    assert second == expected
    assert first["state_version"] == second["state_version"]
    assert second["agvs"]["agv"]["feedback"] == {"state": ["fresh"]}
    assert "defective" not in second["jobs"][job]
    second["agvs"]["agv"]["feedback"]["state"].append("external")
    second["jobs"][job]["risk"] = 99
    second["rankings"]["input"].append("external")
    assert sim.agvs["agv"]["feedback"]["state"] == ["fresh"]
    assert sim.jobs[job]["risk"] == 0.125
    assert sim.rankings["input"] == [job]
    assert protocol.public_view() == expected
    protocol.abort()
    after = protocol.public_view()
    with monkeypatch.context() as patch:
        patch.setattr(production, "detached_copy", copy.deepcopy)
        assert after == protocol.public_view()
    assert after["agvs"]["agv"]["feedback"] is None
    assert after["jobs"][job]["risk"] == before["jobs"][job]["risk"]


def test_snapshot_keeps_extension_hooks_aliases_and_cycles(monkeypatch):
    from test_production_runtime import small_scenario

    import smartsom.engine.production as production
    from smartsom.engine.production import ProductionSimulator

    sim = ProductionSimulator(small_scenario(), contract="v3")
    calls = []

    class Extension:
        def __deepcopy__(self, memo):
            calls.append("extension")
            return {"copied": True}

    cycle = []
    cycle.append(cycle)
    sim.agvs["agv"]["payload"] = [cycle, cycle, Extension()]
    actual = sim.snapshot(public=True)
    assert calls == ["extension"]
    calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(production, "detached_copy", copy.deepcopy)
        expected = sim.snapshot(public=True)
    assert calls == ["extension"]
    a = actual["agvs"]["agv"]["payload"]
    e = expected["agvs"]["agv"]["payload"]
    assert a[0] is a[1] and a[0][0] is a[0]
    assert e[0] is e[1] and e[0][0] is e[0]
    assert a[2] == e[2] == {"copied": True}
    a[0].append("external")
    assert len(cycle) == 1


def test_native_step_descriptors_are_fresh_and_recheck_bypass_mutations():
    from dataclasses import asdict

    from smartsom.domain.production import ProductionStep
    from smartsom.engine.public_projection import step_descriptor

    step = ProductionStep("op", "drill", 3, (("machine", 5),), "7/2")
    first = step_descriptor(step)
    assert first == asdict(step)
    first["operation_type"] = "external"
    object.__setattr__(step, "nominal_ticks", 11)
    second = step_descriptor(step)
    assert second == asdict(step) and second["nominal_ticks"] == 11
    assert first is not second
    object.__setattr__(step, "machine_nominal_ticks", (["machine", 13],))
    third = step_descriptor(step)
    assert third == asdict(step)
    third["machine_nominal_ticks"][0][1] = 99
    assert step.machine_nominal_ticks[0][1] == 13


def test_step_descriptor_fallback_preserves_subclass_fields_and_copy_hooks(monkeypatch):
    from dataclasses import asdict, dataclass

    import smartsom.engine.public_projection as projection
    from smartsom.domain.production import ProductionStep
    from smartsom.engine.public_projection import step_descriptor

    @dataclass(frozen=True)
    class ExtendedStep(ProductionStep):
        extra: object = None

    calls = []

    class Extension:
        def __deepcopy__(self, memo):
            calls.append("copy")
            return {"custom": [1]}

    extended = ExtendedStep("op", "drill", 3, extra=Extension())
    assert step_descriptor(extended) == asdict(extended)
    assert calls == ["copy", "copy"]
    step = ProductionStep("op", "drill", 3)
    original = ProductionStep.__getstate__
    fallbacks = []

    def fallback(value):
        fallbacks.append(value)
        return asdict(value)

    with monkeypatch.context() as patch:
        patch.setattr(projection, "asdict", fallback)
        patch.setattr(ProductionStep, "__getstate__", lambda self: original(self))
        assert step_descriptor(step) == asdict(step)
        assert fallbacks == [step]
    with monkeypatch.context() as patch:
        patch.setitem(copy._deepcopy_dispatch, int, lambda value, memo: value + 10)
        assert step_descriptor(step) == asdict(step)


def test_public_route_projection_updates_same_object_at_same_tick_and_stage(
    monkeypatch,
):
    from test_production_runtime import small_scenario

    import smartsom.engine.production_protocol as protocols
    from smartsom.engine.production import ProductionSimulator

    sim = ProductionSimulator(small_scenario(), contract="v3")
    sim.protocol.begin()
    job = next(iter(sim.jobs))
    first = sim.protocol.public_view()
    demand = sim.demands[sim.jobs[job]["demand"]]
    step = demand.steps[0]
    original = step.nominal_ticks
    try:
        object.__setattr__(step, "nominal_ticks", original + 5)
        actual = sim.protocol.public_view()
        with monkeypatch.context() as patch:
            patch.setattr(protocols, "step_descriptor", protocols.asdict)
            assert actual == sim.protocol.public_view()
        assert actual["state_version"] == first["state_version"]
        assert actual["jobs"][job]["original_steps"][0]["nominal_ticks"] == original + 5
        actual["jobs"][job]["original_steps"][0]["nominal_ticks"] = -1
        assert (
            actual["jobs"][job]["remaining_steps"][0]["nominal_ticks"] == original + 5
        )
        assert (
            sim.protocol.public_view()["jobs"][job]["original_steps"][0][
                "nominal_ticks"
            ]
            == original + 5
        )
    finally:
        object.__setattr__(step, "nominal_ticks", original)
        sim.protocol.abort()


def test_native_step_projection_avoids_recursive_asdict_and_reads_current_fields(
    monkeypatch,
):
    import smartsom.engine.public_projection as projection
    from smartsom.domain.production import ProductionStep

    def unexpected(value):
        raise AssertionError("native scalar descriptor entered recursive asdict")

    step = ProductionStep("op", "drill", 3, (("machine", 5),))
    monkeypatch.setattr(projection, "asdict", unexpected)
    assert projection.step_descriptor(step) == {
        "operation_id": "op",
        "operation_type": "drill",
        "nominal_ticks": 3,
        "machine_nominal_ticks": (("machine", 5),),
        "reference_ticks": None,
    }
    object.__setattr__(step, "nominal_ticks", 9)
    assert projection.step_descriptor(step)["nominal_ticks"] == 9


def test_nested_value_hook_can_replace_its_dict_key_before_key_is_copied():
    key = float("1.25")

    class Extension:
        def __deepcopy__(self, memo):
            memo[id(key)] = "replacement key"
            return {"custom": [1]}

    graph = {"mixed": {key: Extension(), "scalar": 4}, "tail": (key, [key])}
    actual = detached_copy(graph)
    assert actual == copy.deepcopy(graph)
    assert actual["mixed"] == {"replacement key": {"custom": [1]}, "scalar": 4}
    assert actual["tail"] == ("replacement key", ["replacement key"])


def test_mixed_builtin_graph_keeps_repeated_tuple_and_mutable_aliases():
    shared = [1, {"nested": 2}]
    pair = (3, shared, "text")
    graph = {"rows": [pair, pair, shared], "alias": shared, "scalar": 7}
    actual = detached_copy(graph)
    assert actual == copy.deepcopy(graph)
    assert actual["rows"][0] is actual["rows"][1]
    assert actual["rows"][0][1] is actual["rows"][2] is actual["alias"]
    actual["alias"][1]["nested"] = 99
    assert shared[1]["nested"] == 2
