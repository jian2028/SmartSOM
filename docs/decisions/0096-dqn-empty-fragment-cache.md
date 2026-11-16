# 0096 — Normalize empty RLlib fragment caches at replay admission

Status: user-authorized minimal fix; verification in progress.

Ray 2.60 cut/prune/concat can produce an end-reward cache key of zero without
a corresponding pending action. A newly appearing agent at the fragment boundary
then causes official concat to dereference that nonexistent action. A four-step
official episode reproduction requires no SmartSOM simulator or training.

Use the existing replay subclass's public add entry point and official episode
get_state/from_state methods. Remove only zero end-reward cache entries whose
agent has no pending action. Copy the cache rather than modifying incoming
episodes; all observations, actions, recorded rewards, begin caches, termination,
lookback, identities and indices remain unchanged. Nonzero unmatched rewards are
not discarded. This is a pinned dependency on the exported Ray 2.60 state schema,
not a patch to Ray source or a private-method override.

Keep official sampling, priorities, loss, epsilon, agent ordering and episode
boundaries. Do not switch to complete-episode sampling or add synthetic actions.
Test against an uncut trajectory, distinct nonzero rewards, input immutability,
multiple cuts, sampling and checkpoint continuation. Measure the added state
inspection cost; performance acceptance remains pending until paired runs pass.
