"""Lossless storage, ownership and continuation of native DQN replay rows."""

import copy
import os
import pickle
import random
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import pytest

from smartsom.learning.compact_replay import SCHEMA, StoredRow, _Float64Vector
from smartsom.learning.production_collection import Replay


def _bits(value):
    if type(value) is float:
        return float, struct.pack("<d", value)
    if type(value) is dict:
        return dict, tuple((key, _bits(item)) for key, item in value.items())
    if type(value) in (tuple, list):
        return type(value), tuple(_bits(item) for item in value)
    return type(value), value


def _input(index=0):
    return {
        "context": [index / 7, 1.0000000000000002, -0.0, 1e-300],
        "candidates": [[index / 3, 0.25], [1.0, -2.0]],
        "prefix": [],
        "mask": [True, False],
        "feature_width": 2,
    }


def _row(index=0):
    return {
        "input": _input(index),
        "next_input": _input(index + 1),
        "action": index % 2,
        "reward": index / 9,
        "dt": index % 4,
        "terminated": bool(index % 3),
        "choice_decision": bool(index % 2),
    }


class _ListReplay:
    """Independent original slot and RNG reference."""

    def __init__(self, capacity, seed):
        self.capacity, self.rows, self.position = capacity, [], 0
        self.rng = random.Random(seed)
        self.insertions = 0

    def add(self, row):
        self.insertions += 1
        row = copy.deepcopy(dict(row, diagnostic_insertion=self.insertions))
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "rows": self.rows,
            "position": self.position,
            "insertions": self.insertions,
            "random": self.rng.getstate(),
        }


def test_float64_payload_and_public_types_are_bitwise_lossless():
    # Include non-float32 values, both zeros, infinities, and signed NaN payloads.
    values = [
        1.0000000000000002,
        1e-300,
        sys.float_info.max,
        0.0,
        -0.0,
        float("inf"),
        float("-inf"),
        struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0],
        struct.unpack("<d", bytes.fromhex("a50000000000f8ff"))[0],
    ]
    source = _row()
    source["input"]["context"] = values
    source["input"]["candidates"] = (tuple(values), list(values))
    source["input"]["prefix"] = ()
    source["next_input"]["context"] = tuple(values)
    replay = Replay(2, 17)
    replay.add(source)
    stored = replay._rows[0]
    assert stored.compact
    packed = stored.value["input"]["context"]
    assert type(packed) is _Float64Vector
    assert packed.payload == struct.pack(f"<{len(values)}d", *values)
    assert len(packed.payload) == 8 * len(values)
    expected = dict(source, diagnostic_insertion=1)
    assert _bits(replay.rows[0]) == _bits(expected)
    assert _bits(replay.sample(1)[0]) == _bits(expected)


def test_mixed_numbers_and_unknown_fields_are_not_coerced():
    row = _row()
    row["input"]["context"] = [2**100, True, 1.0000000000000002, -0.0]
    row["input"]["feature_width"] = 2**100
    row["input"]["extension"] = {"labels": ["a", None], "vector": [1.0, 2.0]}
    row["extension"] = (False, b"opaque", {"data": [1.0, 2.0]})
    replay = Replay(2, 17)
    replay.add(row)
    assert _bits(replay.rows[0]) == _bits(dict(row, diagnostic_insertion=1))
    assert type(replay._rows[0].value["input"]["context"]) is list
    assert type(replay._rows[0].value["input"]["extension"]["vector"]) is list


def test_compact_ring_retains_fewer_objects_without_original_feature_vectors():
    def observation(index):
        value = _input(index)
        value["context"] = [(index + offset) / 7 for offset in range(1024)]
        return value

    def footprint(value):
        seen, counts = set(), Counter()

        def visit(item):
            if id(item) in seen:
                return 0
            seen.add(id(item))
            counts[type(item)] += 1
            size = sys.getsizeof(item)
            if type(item) is dict:
                size += sum(visit(key) + visit(value) for key, value in item.items())
            elif type(item) in (list, tuple):
                size += sum(visit(value) for value in item)
            elif is_dataclass(item):
                size += sum(visit(getattr(item, field.name)) for field in fields(item))
            return size

        return visit(value), counts, seen

    replay, legacy = Replay(8, 9), _ListReplay(8, 9)
    previous = observation(0)
    for index in range(8):
        current = observation(index + 1)
        row = dict(_row(index), input=previous, next_input=current)
        replay.add(row)
        legacy.add(row)
        previous = current
    compact_bytes, compact_objects, compact_ids = footprint(replay._rows)
    legacy_bytes, legacy_objects, _ = footprint(legacy.rows)
    # Scoped retained Python objects, not RSS or a timing assertion. Adjacent
    # transitions share immutable floats in the original deepcopy reference.
    assert compact_bytes < legacy_bytes * 0.7
    assert legacy_objects[float] > 9000
    assert compact_objects[float] == 8  # Only each transition's scalar reward.
    assert id(previous["context"]) not in compact_ids
    assert all(row.compact for row in replay._rows)
    assert _bits(list(replay.rows)) == _bits(legacy.rows)


@pytest.mark.parametrize("capacity", [1, 3, 37])
def test_overwrite_slots_sampling_rng_and_diagnostic_age_match_legacy(capacity):
    actual, legacy = Replay(capacity, 931), _ListReplay(capacity, 931)
    for index in range(2 * capacity + 5):
        row = _row(index) if index % 2 else {"x": index}
        actual.add(row)
        legacy.add(row)
        assert _bits(list(actual.rows)) == _bits(legacy.rows)
        assert actual.position == legacy.position
        assert actual.insertions == legacy.insertions
        for count in (0, 1, len(legacy.rows) // 2, len(legacy.rows)):
            expected = legacy.rng.sample(legacy.rows, count)
            sampled = actual.sample(count)
            assert _bits(sampled) == _bits(expected)
            assert actual.rng.getstate() == legacy.rng.getstate()
            assert [actual.insertions - r["diagnostic_insertion"] for r in sampled] == [
                legacy.insertions - r["diagnostic_insertion"] for r in expected
            ]
    for count in (-1, capacity + 1):
        before = actual.rng.getstate()
        with pytest.raises(ValueError):
            actual.sample(count)
        assert actual.rng.getstate() == before


@pytest.mark.parametrize("legacy_format", [False, True])
def test_pickle_checkpoint_continues_exactly_across_overwrite(legacy_format):
    original = _ListReplay(7, 981)
    for index in range(19):
        original.add(_row(index))
    original.rng.sample(original.rows, 3)
    replay = Replay(7, 0)
    replay.load_state_dict(original.state_dict())
    state = original.state_dict() if legacy_format else replay.state_dict()
    restored = Replay(7, 999)
    restored.load_state_dict(pickle.loads(pickle.dumps(state, protocol=5)))
    for index in range(19, 34):
        for target in (original, replay, restored):
            target.add(_row(index))
        expected = original.rng.sample(original.rows, 5)
        assert _bits(replay.sample(5)) == _bits(expected)
        assert _bits(restored.sample(5)) == _bits(expected)
        assert replay.position == restored.position == original.position
        assert replay.insertions == restored.insertions == original.insertions
        assert (
            replay.rng.getstate() == restored.rng.getstate() == original.rng.getstate()
        )


def test_old_checkpoint_without_insertion_counter_remains_compatible():
    legacy = _ListReplay(3, 981)
    for index in range(3):
        legacy.add({"x": index})
    state = legacy.state_dict()
    del state["insertions"]
    for row in state["rows"]:
        del row["diagnostic_insertion"]
    replay = Replay(3, 0)
    replay.load_state_dict(state)
    assert replay.insertions == 0
    assert replay.sample(3) == legacy.rng.sample(legacy.rows, 3)
    replay.add({"x": 4})
    restored = Replay(3, 1)
    restored.load_state_dict(replay.state_dict())
    assert restored.sample(3) == replay.sample(3)


def test_producer_sample_rows_checkpoint_and_restores_are_isolated():
    producer = _row()
    replay = Replay(4, 3)
    replay.add(producer)
    replay.add(producer)
    expected = _bits(list(replay.rows))
    producer["input"]["context"][0] = 444.0
    producer["next_input"]["mask"][0] = False
    sampled = replay.sample(2)
    sampled[0]["input"]["context"][0] = 555.0
    sampled[0]["input"]["mask"][0] = False
    assert sampled[1]["input"]["context"][0] == 0.0
    assert sampled[1]["input"]["mask"][0] is True
    public = replay.rows[0]
    public["input"]["context"].clear()
    public_slice = replay.rows[:]
    public_slice[0]["next_input"]["mask"].clear()
    assert len(replay.rows) == 2
    assert _bits(list(replay.rows)) == expected
    state = replay.state_dict()
    assert state["schema"] == SCHEMA
    left, right = Replay(4, 0), Replay(4, 0)
    left.load_state_dict(state)
    right.load_state_dict(state)
    state["rows"][0].value["input"]["mask"][0] = False
    state["rows"].clear()
    left._rows[0].value["input"]["mask"][0] = False
    assert right.rows[0]["input"]["mask"][0] is True
    assert _bits(list(replay.rows)) == expected
    assert _bits(list(right.rows)) == expected
    # A ring mutation cannot modify an already returned checkpoint either.
    detached = replay.state_dict()
    replay.add(_row(8))
    assert len(detached["rows"]) == 2


@dataclass
class _Extension:
    values: list


def test_custom_objects_and_shared_or_cyclic_graphs_use_owned_fallback():
    shared = [1.0, -0.0]
    cycle = []
    cycle.append(cycle)
    rows = [
        dict(_row(), extension=_Extension([4])),
        {"input": {"context": shared}, "extension": shared},
        {"cycle": cycle},
    ]
    replay = Replay(3, 23)
    for row in rows:
        replay.add(row)
    assert all(not row.compact for row in replay._rows)
    values = list(replay.rows)
    assert type(values[0]["extension"]) is _Extension
    assert values[0]["extension"] is not rows[0]["extension"]
    assert values[1]["input"]["context"] is values[1]["extension"]
    assert values[1]["extension"] is not shared
    assert values[2]["cycle"][0] is values[2]["cycle"]
    assert values[2]["cycle"] is not cycle
    values[0]["extension"].values.append(5)
    values[1]["extension"].append(6)
    values[2]["cycle"].append(7)
    restored = Replay(3, 1)
    restored.load_state_dict(pickle.loads(pickle.dumps(replay.state_dict())))
    assert restored.rows[0]["extension"].values == [4]
    assert restored.rows[1]["extension"] == [1.0, -0.0]
    assert len(restored.rows[2]["cycle"]) == 1


def test_numpy_types_are_fallback_and_tensor_input_dtypes_are_unchanged():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from smartsom.learning.production_models import tensor_inputs

    row = _row()
    row["input"]["context"][0] = 1.0000000000000002
    replay = Replay(2, 3)
    replay.add(row)
    expected = tensor_inputs([row["input"]])
    decoded = tensor_inputs([replay.rows[0]["input"]])
    for field, value in expected.items():
        assert value.dtype == decoded[field].dtype
        assert value.shape == decoded[field].shape
        assert value.numpy().tobytes() == decoded[field].numpy().tobytes()
    assert expected["context"].numpy().dtype == np.float64
    array = np.asarray([1.0000000000000002, -0.0], dtype=">f8")
    source = dict(_row(), extension=array, scalar=np.float32(1.5))
    replay.add(source)
    assert not replay._rows[1].compact
    array[0] = 9.0
    restored = replay.rows[1]
    assert type(restored["scalar"]) is np.float32
    assert restored["extension"].dtype == np.dtype(">f8")
    assert restored["extension"].tobytes() == struct.pack(
        ">2d", 1.0000000000000002, -0.0
    )
    restored["extension"][0] = 8.0
    assert replay.rows[1]["extension"][0] == 1.0000000000000002


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema", "smartsom.compact-replay/v999"),
        ("capacity", 4),
        ("capacity", True),
        ("rows", None),
        ("rows", [{}, {}, {}, {}]),
        ("rows", [{}]),
        ("position", -1),
        ("position", 3),
        ("position", True),
        ("position", 0),
        ("insertions", -1),
        ("insertions", 1.5),
        ("random", (3, (), None)),
    ],
)
def test_malformed_state_is_rejected_without_changing_live_replay(field, invalid):
    replay = Replay(3, 23)
    replay.add(_row())
    before = pickle.dumps(replay.state_dict())
    broken = replay.state_dict()
    broken[field] = invalid
    with pytest.raises(ValueError):
        replay.load_state_dict(broken)
    assert pickle.dumps(replay.state_dict()) == before


@pytest.mark.parametrize("invalid", [b"", b"1234567", bytearray(8)])
def test_malformed_float64_payload_is_rejected_atomically(invalid):
    replay = Replay(3, 23)
    replay.add(_row())
    before = pickle.dumps(replay.state_dict())
    broken = replay.state_dict()
    vector = broken["rows"][0].value["input"]["context"]
    broken["rows"][0].value["input"]["context"] = replace(vector, payload=invalid)
    with pytest.raises(ValueError, match="float64"):
        replay.load_state_dict(broken)
    assert pickle.dumps(replay.state_dict()) == before


def test_malformed_container_and_legacy_row_are_rejected():
    replay = Replay(3, 23)
    replay.add(_row())
    broken = replay.state_dict()
    broken["rows"][0] = StoredRow([], True)
    with pytest.raises(ValueError, match="row"):
        replay.load_state_dict(broken)
    broken = replay.state_dict()
    broken["rows"][0].value["cycle"] = broken["rows"][0].value
    with pytest.raises(ValueError, match="container"):
        replay.load_state_dict(broken)
    broken = replay.state_dict()
    del broken["schema"]
    with pytest.raises(ValueError, match="legacy"):
        replay.load_state_dict(broken)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_requires_a_positive_integer(capacity):
    with pytest.raises(ValueError, match="capacity"):
        Replay(capacity, 0)


def test_base_import_and_generic_replay_require_no_optional_frameworks():
    source = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'ray'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import smartsom
from smartsom.learning.production_collection import Replay
value = Replay(2, 1)
value.add({'x': 1})
assert value.sample(1)[0]['x'] == 1
"""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
