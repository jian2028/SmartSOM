"""Immutable, pickleable named facts for observation authors."""

from collections.abc import Mapping

from smartsom.domain.observation_data import BaseInputs


class Facts(Mapping):
    """Immutable named values with constant-time lookup and portable pickling."""

    __slots__ = ("entries", "_index")

    def __init__(self, entries):
        entries = tuple(entries)
        if any(not isinstance(k, str) for k, _ in entries):
            raise TypeError("named observation keys must be strings")
        object.__setattr__(self, "entries", tuple((k, freeze(v)) for k, v in entries))
        object.__setattr__(self, "_index", dict(self.entries))

    def __setattr__(self, name, value):
        raise AttributeError("public facts are immutable")

    def __getitem__(self, key):
        return self._index[key]

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self.entries)

    def __reduce__(self):
        return type(self), (self.entries,)


def freeze(value):
    # Public snapshots are mostly builtin scalars and containers. Avoid ABC
    # instance checks for every scalar while retaining custom Mapping support.
    kind = type(value)
    if kind is float or kind is int or kind is str or value is None or kind is bool:
        return value
    if kind is dict:
        return Facts(value.items())
    if kind is list or kind is tuple:
        return tuple(freeze(v) for v in value)
    if isinstance(value, (Facts, BaseInputs)):
        return value
    if isinstance(value, Mapping):
        return Facts(value.items())
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
