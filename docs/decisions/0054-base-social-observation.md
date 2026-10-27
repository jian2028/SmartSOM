# 0054 — Optional candidate-linked social observation

Date: 2026-10-27
Status: User-authorized implementation; performance experiments remain separate.

For explicitly selected social arms only, this supersedes ADR0053's actor
observation/network scope with the increment in [the SL specification](../social-learning.md).
ADR0053's default Base and all physical, action, reward, Mover and critic contracts
remain unchanged. The earlier reconstructed-input option remains a common Base
choice, never an implicit side effect of enabling SL.

Add detached semantic candidate-linked public peer sets and optional actor-only
conditional attention, with a same-input fixed-pooling control. Keep five policy
groups; split Dispatcher intervention selectors by stage. Current-state peers
are related by workflow capability, local stock or effective reservations. They
do not expose hidden outcomes, future calendars, unaccepted intent or grant new
service/routing authority. No history or reward shaping is added.

Version SL checkpoints separately. Preserve existing Base initialization and
checkpoint loading. Frozen experiments remain bound to their original sources;
this branch is not a hot patch or an authorization to start formal experiments.
