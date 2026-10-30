# 0061 — Network ownership, fixed cases and research packages

Status: accepted; implementation acceptance is recorded separately.

Supersedes the role-only parameter ownership and episode-only validation boundary
in ADR0059, and the authoring/output layout portions of ADR0058. ADR0060's metric,
missing-data and evaluation protocols remain in effect. No formal run is authorized.

A learning policy selects shared or independent ownership per role. Independence
includes actor/critic/encoder/optimizer (or Q/target/replay/optimizer), keyed by
stable machine, vehicle or Buffer identity. Independent instances copy the role's
initial template. Role hyperparameters override common learner parameters; entity
specific configuration and partially shared actor/critic combinations are excluded.
PPO advantages retain role-wide normalization before partitioning updates.
DQN shared roles retain the global optimizer-update target clock, with each
role’s configured interval; independent owners use their own update count.
The all-shared default retains its existing numerical path. Custom learners must
advertise ownership and role-parameter capabilities before compilation accepts them.

Training seed counts expand from 101; explicit unique lists remain supported.
Validation and test panels are distinct, method/seed-independent physical worlds.
Imported worlds preserve their realization, not just a random seed label. Compact
historical case recipes carry expected world digests; compilation checks them and
freezes full worlds. Test evidence never selects best checkpoints.

Validation nodes use episodes or exact cumulative training physical ticks.
Tick nodes close the rollout, commit a full update, validate and start a new world.
Truncation retains PPO bootstrap and omits unresolved DQN tails. A node does not
become a successful physical termination. Existing safe-update recovery remains.

Top-level experiments holds topic/study/variant/experiment.yaml. Configs holds
shared inputs and operation profiles. Runs mirrors a concrete study, with
trials/variant/seed-N, local evidence and one report. Only canonical experiment
entries are discovered; auxiliary YAML is never executed. Scientific changes need
a new study. Existing outputs cannot be overwritten. Dates and platforms are
metadata, not the default directory hierarchy.

One host runs a batch. A complete stopped bundle can move to another compatible
host with matching code and dependencies; actual platforms are recorded and
bitwise cross-platform equivalence is not promised. This is not a distributed
scheduler or a cross-host result collector. Historic runs remain in place and
continue only with their original source and environment.
