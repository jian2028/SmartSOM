# 0046 — One maintained compact public-fields interface

Date: 2026-10-13
Status: User-authorized implementation; verification and integration pending

Supersedes ADR0045's dual-version transition and ADR0041's equal-width planes.
The user selected retirement of the wide public-fields encoder/network, stopped
the old local Batch05 run and requested fresh initialization before restarting.

`builtin.public_fields` and `builtin.scoped_candidate` have one compact
implementation, selected by name without a user version field. Frozen packages
retain an internal `compact` identity and strict source/shape/algorithm checks.
Explicit historical version 1/2 selectors and old weights fail closed, rather
than being silently converted. Historical evidence can be reproduced with its
original source commit, not an additional implementation in the current tree.

DQN stores actor/Q context only and computes no critic-only fields. PPO stores
ordered compact actor and critic context and sizes each branch independently.
The value branch has no candidate/prefix placeholders or prefix GRU. Existing
component semantics, legal actions, prefix ordering, rewards, optimizer settings,
training budgets and the shared batch entry remain unchanged. Other distinct
historical policy contracts are outside this scoped retirement.

New local pre-integration runs record the exact committed feature source and
remain development evidence until integrated-main or explicit-tag qualification.
Smaller inputs alone do not establish an end-to-end throughput improvement.
