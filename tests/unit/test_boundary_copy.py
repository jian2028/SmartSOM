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
