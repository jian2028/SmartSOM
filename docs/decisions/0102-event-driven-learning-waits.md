# 0102 — Event-driven learned waits

Status: user-authorized; supersedes ADR0090's ordinary WAIT cadence.

The physical simulator and its same-tick production negotiation order remain
unchanged. RLlib does not require every active entity to act at every physical
tick: an absent observation leaves that agent's action pending. Generating WAIT
for every previously observed entity at every tick unnecessarily multiplied
observations and retained trajectories.

After exhausting a negotiation interval, return one ordinary WAIT observation
only for each semantic agent that made a production decision in that interval.
An accepted WAIT remains pending until the next production observation or the
final observation. Advance physical intervals without decisions internally,
accumulating every interval's reward. All previously started agents receive that
accumulated team reward; a newly observed agent receives no retroactive reward.
The simulator still advances every physical interval, including traffic, service,
quality and failures. No production candidate, reservation or decision is skipped.

At termination or truncation, every started agent still receives a fresh final
observation with one legal WAIT. Ordinary waits remain training states; final
bootstrap states are not invented only at the cutoff. Official PPO/DQN losses,
discounts, exploration and replay remain unchanged. Fewer agent transitions
change discounting, sample budgets and potentially training outcomes, as explicitly
accepted by the user. This does not promise old-policy numerical equivalence or
mid-simulation checkpoint continuation. Existing frozen runs are not rewritten.

The drain uses RLlib's whole-environment `truncateds["__all__"]` signal, which
propagates to each agent. In Ray 2.60, additionally marking every individual
agent truncated triggers its mixed-done inference and also marks the trajectory
terminated, suppressing bootstrap. End-to-end episode and replay checks must
verify truncation without termination, not merely the presence of truncation.

Validation requires physical trace equivalence under the same semantic actions,
complete and truncated costs, official asynchronous reward accumulation, legal
bootstrap masks, real PPO/DQN updates, official checkpoint/replay restoration and
continued training. Performance comparisons retain the same world and completed
work, and report changed agent-step counts separately. Inference-only WAIT
ablation is diagnostic evidence, not proof of training correctness.
