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
            patch.setattr(protocol, "_public_step", protocol.asdict)
            baseline_record = baseline.tick()
        assert fast_record == baseline_record
        if fast_record["decisions"]:
            saw_decisions = True
            saved = copy.deepcopy(fast_record)
            fast.records[0]["observation"]["tick"] = -1
            assert fast_record == saved
    assert saw_decisions
    assert fast.sim.snapshot() == baseline.sim.snapshot()


def test_a_value_copy_hook_can_replace_its_own_scalar_dictionary_key():
    key = float("3.25")

    class Extension:
        def __deepcopy__(self, memo):
            memo[id(key)] = "replacement"
            return "copied"

    graph = {key: Extension()}
    assert detached_copy(graph) == copy.deepcopy(graph) == {"replacement": "copied"}


def test_public_step_serialization_preserves_types_and_detachment():
    from smartsom.domain.production import ProductionStep
    from smartsom.engine.production_protocol import _public_step, asdict

    step = ProductionStep("step", "op", 3, (("machine", 5),), "7/2")
    first = _public_step(step)
    second = _public_step(step)
    assert first == asdict(step)
    assert type(first["machine_nominal_ticks"]) is tuple
    first["operation_id"] = "changed"
    assert second == asdict(step)

    @dataclass(frozen=True, slots=True)
    class ExtendedStep(ProductionStep):
        notes: list | None = None

    extended = ExtendedStep("extended", "op", 2, notes=["note"])
    row = _public_step(extended)
    assert row == asdict(extended)
    row["notes"].append("changed")
    assert extended.notes == ["note"]


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
