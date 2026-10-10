"""Owned replay storage for native Python observations, without optional imports.

Only homogeneous Python-float vectors in the established observation fields are
packed. Their little-endian binary64 payload keeps every bit (including signed
zero and NaN payloads); public reads reconstruct the original list/tuple types.
Unfamiliar objects or aliased/cyclic container graphs use ordinary deepcopy.
"""

import copy
import struct
from collections.abc import Sequence
from dataclasses import dataclass

SCHEMA = "smartsom.compact-replay/v1"
_SCALARS = (type(None), bool, int, float, str, bytes)
_FEATURES = frozenset(("context", "candidates", "prefix"))


@dataclass(frozen=True, slots=True)
class _Float64Vector:
    payload: bytes
    is_tuple: bool

    def decode(self):
        values = struct.unpack(f"<{len(self.payload) // 8}d", self.payload)
        return values if self.is_tuple else list(values)


class _Fallback(Exception):
    pass


def _compact(value, seen, *, path=()):
    kind = type(value)
    if kind in _SCALARS:
        return value
    if kind not in (dict, list, tuple) or id(value) in seen:
        raise _Fallback
    seen.add(id(value))
    if kind is dict:
        if any(type(key) not in _SCALARS for key in value):
            raise _Fallback
        return {
            key: _compact(item, seen, path=(*path, key)) for key, item in value.items()
        }
    if (
        len(path) >= 2
        and path[0] in ("input", "next_input")
        and path[1] in _FEATURES
        and value
        and all(type(item) is float for item in value)
    ):
        return _Float64Vector(struct.pack(f"<{len(value)}d", *value), kind is tuple)
    items = [_compact(item, seen, path=path) for item in value]
    return tuple(items) if kind is tuple else items


def _expand(value):
    if type(value) is _Float64Vector:
        return value.decode()
    if type(value) is dict:
        return {key: _expand(item) for key, item in value.items()}
    if type(value) is list:
        return [_expand(item) for item in value]
    if type(value) is tuple:
        return tuple(_expand(item) for item in value)
    return value


def _validate_compact(value, seen):
    kind = type(value)
    if kind in _SCALARS:
        return
    if kind is _Float64Vector:
        if (
            type(value.payload) is not bytes
            or not value.payload
            or len(value.payload) % 8
            or type(value.is_tuple) is not bool
        ):
            raise ValueError("invalid compact replay float64 vector")
        return
    if kind not in (dict, list, tuple) or id(value) in seen:
        raise ValueError("invalid compact replay container")
    seen.add(id(value))
    if kind is dict:
        if any(type(key) not in _SCALARS for key in value):
            raise ValueError("invalid compact replay dictionary key")
        value = value.values()
    for item in value:
        _validate_compact(item, seen)


@dataclass(frozen=True, slots=True)
class StoredRow:
    """Private storage record; only decode() exposes a mutable observation."""

    value: dict
    compact: bool

    @classmethod
    def encode(cls, row):
        try:
            return cls(_compact(row, set()), True)
        except (_Fallback, RecursionError):
            return cls(copy.deepcopy(row), False)

    def decode(self):
        return _expand(self.value) if self.compact else copy.deepcopy(self.value)

    def validate(self):
        if type(self.value) is not dict or type(self.compact) is not bool:
            raise ValueError("invalid compact replay row")
        if self.compact:
            try:
                _validate_compact(self.value, set())
            except RecursionError as error:
                raise ValueError("invalid compact replay nesting") from error


class ReplayRows(Sequence):
    """Read-only slot-order view with detached rows and constant-time length."""

    def __init__(self, replay):
        self._replay = replay

    def __len__(self):
        return len(self._replay._rows)

    def __getitem__(self, index):
        selected = self._replay._rows[index]
        if isinstance(index, slice):
            return [row.decode() for row in selected]
        return selected.decode()

    def __iter__(self):
        return (row.decode() for row in self._replay._rows)
