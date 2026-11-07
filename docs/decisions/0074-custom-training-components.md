# 0074 — Common reward, critic and centralized learning contracts

Status: implementation; verification is reported separately.

Supersedes ADR0073's centralized-DRL exclusion and the fixed critic/reward wiring
of ADR0070. One physical simulator and the existing legal tasks/mover remain.
No historical E1a dispatch protocol, RLlib runtime or yield mover is imported.

Observation and critic_observation independently project public facts. Critic
contexts identify tick, role, owner and bootstrap, with no fabricated candidates.
Named critic networks implement value(inputs); vector critics retain the explicit
vector value contract. Double DQN never requests critic observations.

A reward component receives committed physical intervals, actions, public views
and separate committed outcome statistics. Reward-only outcomes must not flow into
observations. The core retains the standard diagnostic objective through the
standalone default reward function; selected training objectives are evaluated by
the component runtime. This preserves direct simulator consumers and physical
metrics without embedding experiment-specific reward weights in physics.
Team, role and semantic-agent terms add; parameter sharing does not merge trajectories.
Snapshots cover the state after zero-time choices and before physical advance, and
the committed resulting state. Stateless objectives can opt out of snapshots/events.

Native PPO and Double DQN support one central/factory subject as well as the five
reservation roles. Central actions are semantic task IDs. GAE chains each subject's
successive decisions; gamma discounts physical time and lambda discounts decision
transitions, including zero-time transitions. episode_return preserves defaults.
Terminal bootstrap is zero; truncation uses critic context or a read-only central
candidate snapshot. Unresolved reservation DQN tails retain their prior omission.
Custom learners can use learning.transitions.decision_transitions without core access.

E1a-current is an adaptation: existing process/transport/ADVANCE tasks, shared mover,
64x64 candidate scorer/value networks, gamma=1, GAE lambda=.95 and complete-episode
updates. It is not reproduction of remote E1a results or fixed-256-tick updates.
Source: Morris-11012026 at 3b957785e572d002a608468430e195f64141bc84,
reports/2026-11-05/notes__e1a_summary/e1a_summary.md and configs/algorithms/central01_v2_e1a.yaml.

Checkpoint v4 includes reward state and scientific component identity; old identities
fail explicitly. Complete-update recovery remains the only durable boundary. Existing
runs use their frozen code. The pre-change dirty tree is externally archived at
/Users/jianni/SmartSOM-archives/20261106-e1a-components; source.tar.gz SHA256
59167213fc712fc42fd789369870eeeae91b0fec1d0a517685439ac25c483885.
No commits, publication or formal experiments are authorized by this decision.
