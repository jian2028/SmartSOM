"""Complete per-resource transitions; unresolved safety-cut tails are omitted."""

import pickle
import zlib
from collections.abc import Sequence

from smartsom.domain.base_decisions import GROUPS
from smartsom.learning.subjects import subject_key


class PackedRows(Sequence):
    """Owned DQN rows with immutable value semantics, not cross-read identity.

    Complete row pickle graphs retain all fields and numeric precision. Builtin
    learners use this storage; custom components retain ordinary list storage.
    """

    def __init__(self, rows=()):
        if isinstance(rows, PackedRows):
            # Own the byte sequence without decoding or mutating borrowed state.
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
        return pickle.loads(zlib.decompress(entry))

    def append(self, row):
        self.entries.append(zlib.compress(pickle.dumps(row, protocol=5), level=1))

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


def _transition_rows(
    records,
    rewards,
    *,
    terminal,
    scale,
    gamma=1.0,
    split_stages=(),
    subject_counts=None,
):
    from smartsom.domain.training import subject_rewards

    scales = scale if isinstance(scale, dict) else dict.fromkeys(GROUPS, scale)
    gammas = gamma if isinstance(gamma, dict) else dict.fromkeys(scales, gamma)
    terminals = (
        terminal if isinstance(terminal, dict) else dict.fromkeys(scales, terminal)
    )
    suffixes = {}
    following = {
        subject_key(r, split_stages): r
        for r in getattr(rewards, "final_requests", {}).values()
        if not terminals[r.group]
    }
    streams = {}
    for row in reversed(records):
        request = row["request"]
        factor, discount = scales[request.group], gammas[request.group]
        key = subject_key(request, split_stages)
        if key not in streams:
            streams[key] = subject_rewards(rewards, request)
        stream = streams[key]
        cache_key = (factor, discount, key if stream is not rewards else None)
        if cache_key not in suffixes:
            suffix = [0.0] * (len(stream) + 1)
            for i in reversed(range(len(stream))):
                suffix[i] = stream[i] * factor + discount * suffix[i + 1]
            suffixes[cache_key] = suffix
        suffix = suffixes[cache_key]
        next_request = following.get(key)
        if subject_counts is not None:
            counts = subject_counts.setdefault(key, {"observed": 0, "omitted": 0})
            counts["observed"] += 1
            counts["omitted"] += int(
                next_request is None and not terminals[request.group]
            )
        if next_request is None and not terminals[request.group]:
            yield request.group, None
        else:
            end = next_request.tick if next_request else len(rewards)
            yield (
                request.group,
                {
                    "request": request,
                    "index": row["index"],
                    "reward": suffix[request.tick]
                    - discount ** (end - request.tick) * suffix[end],
                    "next": next_request,
                    "physical_dt": end - request.tick,
                },
            )
        following[key] = request


def transitions(
    records,
    rewards,
    *,
    terminal,
    scale,
    gamma=1.0,
    split_stages=(),
    subject_counts=None,
):
    pending = {g: [] for g in (scale if isinstance(scale, dict) else GROUPS)}
    truncated_tails = 0
    for group, row in _transition_rows(
        records,
        rewards,
        terminal=terminal,
        scale=scale,
        gamma=gamma,
        split_stages=split_stages,
        subject_counts=subject_counts,
    ):
        if row is None:
            truncated_tails += 1
        else:
            pending[group].append(row)
    return pending, truncated_tails


def bounded_transitions(
    records,
    rewards,
    *,
    terminal,
    scale,
    ownership,
    gamma=1.0,
    split_stages=(),
    subject_counts=None,
):
    pending = {key: PackedRows() for keys in ownership.keys.values() for key in keys}
    counts = dict.fromkeys(pending, 0)
    truncated_tails = 0
    for group, row in _transition_rows(
        records,
        rewards,
        terminal=terminal,
        scale=scale,
        gamma=gamma,
        split_stages=split_stages,
        subject_counts=subject_counts,
    ):
        if row is None:
            truncated_tails += 1
            continue
        key = ownership.key(row["request"])
        counts[key] += 1
        if len(pending[key]) < ownership.options[group].replay_capacity:
            pending[key].append(row)
    return pending, truncated_tails, counts
