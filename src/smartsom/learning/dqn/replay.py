"""Complete per-resource transitions; unresolved safety-cut tails are omitted."""

import pickle
import zlib
from collections.abc import Sequence

from smartsom.domain.base_decisions import GROUPS, BaseCandidate, BaseDecision
from smartsom.domain.social import SocialCandidate, SocialLink

_ATOMS = {str, int, float, bool, bytes, type(None)}
_FIELDS = {
    cls: tuple(cls.__dataclass_fields__)
    for cls in (BaseCandidate, BaseDecision, SocialCandidate, SocialLink)
}


def immutable_graph(value):
    """Strict built-in value graph; unknown/mutable extensions retain raw rows."""
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind in _ATOMS:
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if kind is tuple:
            # Most data are scalar feature vectors; avoid a Python stack entry
            # for each scalar while still verifying every value's exact type.
            if not all(type(v) in _ATOMS for v in item):
                pending.extend(item)
        elif kind in _FIELDS:
            pending.extend(getattr(item, name) for name in _FIELDS[kind])
        else:
            return False
    return True


class PackedRows(Sequence):
    """Owned DQN rows with immutable value semantics, not cross-read identity.

    Complete row pickle graphs retain all fields and numeric precision. Unknown
    mutable values remain raw; no capacity or scientific setting is changed.
    """

    def __init__(self, rows=()):
        if isinstance(rows, PackedRows):
            # Entries are immutable bytes or compatibility raw rows. Own the
            # sequence container without decoding or mutating a borrowed state.
            self.entries = rows.entries.copy()
            return
        self.entries = []
        for row in rows:
            self.append(row)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        entry = self.entries[index]
        return pickle.loads(zlib.decompress(entry)) if type(entry) is bytes else entry

    def append(self, row):
        if type(row) is dict and immutable_graph(tuple(row.items())):
            row = zlib.compress(pickle.dumps(row, protocol=5), level=1)
        self.entries.append(row)

    def merge_newest(self, newest_first, capacity):
        self.entries.extend(reversed(newest_first.entries))
        del self.entries[:-capacity]


class CheckpointReplay(dict):
    """Serialize complete CPU request graphs once, without Torch tensor probing.

    The reducer reconstructs an ordinary dict using only the standard library,
    so existing checkpoint readers need no new class or storage-format branch.
    No field, precision, order or shared request/next reference is discarded.
    """

    def __reduce_ex__(self, protocol):
        return pickle.loads, (pickle.dumps(dict(self), protocol=5),)


def _transition_rows(records, rewards, *, terminal, scale):
    scales = scale if isinstance(scale, dict) else dict.fromkeys(GROUPS, scale)
    suffixes = {}
    for factor in set(scales.values()):
        suffix = [0.0] * (len(rewards) + 1)
        for i in reversed(range(len(rewards))):
            suffix[i] = suffix[i + 1] + rewards[i] * factor
        suffixes[factor] = suffix
    following = {}
    for row in reversed(records):
        request = row["request"]
        suffix = suffixes[scales[request.group]]
        key = (request.group, request.owner)
        next_request = following.get(key)
        if next_request is None and not terminal:
            yield request.group, None
        else:
            end = next_request.tick if next_request else len(rewards)
            yield (
                request.group,
                {
                    "request": request,
                    "index": row["index"],
                    "reward": suffix[request.tick] - suffix[end],
                    "next": next_request,
                    "physical_dt": end - request.tick,
                },
            )
        following[key] = request


def transitions(records, rewards, *, terminal, scale):
    pending = {g: [] for g in GROUPS}
    truncated_tails = 0
    for group, row in _transition_rows(
        records, rewards, terminal=terminal, scale=scale
    ):
        if row is None:
            truncated_tails += 1
        else:
            pending[group].append(row)
    return pending, truncated_tails


def bounded_transitions(records, rewards, *, terminal, scale, ownership):
    pending = {key: PackedRows() for keys in ownership.keys.values() for key in keys}
    counts = dict.fromkeys(pending, 0)
    truncated_tails = 0
    for group, row in _transition_rows(
        records, rewards, terminal=terminal, scale=scale
    ):
        if row is None:
            truncated_tails += 1
            continue
        key = ownership.key(row["request"])
        counts[key] += 1
        if len(pending[key]) < ownership.options[group].replay_capacity:
            pending[key].append(row)
    return pending, truncated_tails, counts
