# 0090 — Explicit learned waiting states at physical boundaries

Status: user-authorized; adapter implementation and RLlib acceptance in progress.

This supersedes ADR0089's restriction to observations only at actual production
decisions. A real drain cutoff can interrupt an agent after a production action
while it is still busy. Ray 2.58 requires a final observation for a truncated
agent; masked DQN also requires a nonempty next-action set for bootstrap.

After exhausting the existing same-tick negotiation generator, expose a waiting
frame for previously observed semantic agents. Its only legal action is WAIT.
Accepting these waits advances exactly the physical interval that the existing
shared Mover and protocol would have advanced. Never insert another physical
interval, reorder production decisions, or replace an illegal production action.

Waiting states must be present in ordinary training, not invented only at the
cutoff. They contribute official agent decision steps and therefore affect
official discounting. The user explicitly accepted this modeling change.
Construct waiting observations from current permitted local information; do not
reuse a stale production observation or expose privileged reward information.
PPO's separately declared global critic input remains isolated from its actor.

At a drain cutoff return current waiting observations with one legal WAIT for
all active agents, `truncated=True`, and no complete J. Do not execute those final
actions. Complete delivery terminates the task and retains the complete objective.
Record task cost once per physical interval, independently of replicated training
rewards and zero-time action calls.

Acceptance requires physical trace equivalence under matched semantic production
actions, unchanged final factory state, legal masks in ordinary and truncated
waiting states, official PPO/DQN learning through waits, correct bootstrap flags,
and a full performance comparison that includes the additional agent steps.
