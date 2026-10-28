"""Immutable, pickleable named facts for observation authors."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Facts(Mapping):
    entries: tuple

    def __getitem__(self, key):
        for name, value in self.entries:
            if key == name:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (name for name, _ in self.entries)

    def __len__(self):
        return len(self.entries)


def freeze(value):
    if isinstance(value, Mapping):
        return Facts(tuple((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("observation inputs must contain only named public data")


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value
