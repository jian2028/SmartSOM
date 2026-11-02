# 0068 — Lossless resident storage for built-in DQN rows

Status: proposed in an isolated candidate; independent review, resource and
end-to-end performance acceptance remain pending. No formal experiment is
started or historical model migrated by this decision.

## Context

The official DQN retains full request graphs in records and replay. The
checkpoint-only pickle optimization reduces serialization traversal, not resident
memory. A conditional synthetic envelope using maximal existing 128-tick request
graphs, 10,000 replay rows per role and 300,000 records exceeds an 8 GiB process
budget even when only containers and dataclass instances are counted. This does
not establish the actual memory use of every future world. Existing raw replay
and archived rule-DQN compact storage are distinct schemas.

## Decision

For built-in Base/social observations and encoders without a custom complete
network, store owned records and replay rows as complete protocol-5 pickle
graphs compressed with zlib level 1. A strict recursive exact-type check permits
only known frozen request/candidate/social dataclasses, tuples and immutable
scalars. Unknown or mutable values retain their original raw row. Custom
observation/encoder/network configurations retain raw storage throughout.

Every original field is retained, including inputs and rule_audit when present;
there is no quantization, capacity reduction or delayed threshold-based packing.
The record count still follows the unchanged rollout/episode boundary. Packed
records consume bytes proportional to their count; there is no new truncation
or claim of a universal episode-count bound.

Packed reads preserve values and within-row graph structure, not object identity
across reads or across separate row blobs. The supported built-in DQN consumers
read immutable request values and do not depend on cross-row identity. This
decision does not extend to PPO, arbitrary custom consumers or mutable graphs.

Generate transitions in reverse record order with the existing suffix reward
arithmetic and next-request-per-owner relation. Retain only the newest replay
capacity transitions per ownership key, in packed form; count all transitions
and omitted tails even when older rows cannot survive replay eviction. Merge
the retained rows into replay in the same chronological order as before.
Sampling indices/order, RNG use, batch tensors, learning math, gradient clipping,
Adam state, target cadence and scientific settings remain unchanged.

## Checkpoints and compatibility

New checkpoint payloads may contain the packed sequence implementation and
therefore require the new reader. They are not backward-readable by archived
source that lacks the class or expects mutable raw lists. Source-bound
continuation rules remain in force; old scientific inputs/weights are never
silently reinterpreted. Existing compatible official raw-list checkpoints can
be loaded by the new reader without modifying borrowed payloads or lists.
Packed input copies only the sequence container, sharing immutable byte entries;
raw input is encoded non-destructively. Legacy raw deserialization and conversion
peaks remain a separate limit; arbitrarily large historical raw checkpoints are
not guaranteed to fit the process budget.

The original complete-update checkpoint boundary remains unchanged. This does
not add pending rollout records to checkpoint state or turn interrupted partial
rollouts into resumable transactions. Existing model, target, optimizer, counter,
RNG and checkpoint identity behavior remain authoritative.

## Acceptance and limitations

Require raw/candidate equality on the complete approved 128-tick engineering
trajectory, packed inputs, actions, rewards, losses, model/Adam/target tensors,
counters, RNG, restore and next update. Add focused overflow-order, truncated-tail,
custom/mutable fallback and compatible raw-checkpoint tests. Independently review
the implementation before integration.

Resource acceptance must use distinct compressed byte allocations under the
unchanged 8 GiB process guard and conservative available-RAM reserve, measuring
records plus full replay, streamed update and save/load peaks. Small-probe RSS
and encoded-size extrapolations are not full-load evidence. Fallback raw rows
and legacy load peaks require explicit accounting.

Performance acceptance uses the old-custom versus new-official baseline. The
latest requested gate is five paired 1280-tick measurements for each Base/ALL
arm, comparing all post-start work including final save/close within 5%.
One-time startup and total elapsed time are reported separately. Short core and
codec timings, including earlier 128-tick probes, are diagnostic only. No
thresholds or delayed packing may be introduced to game the bounded sample.
