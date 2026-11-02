# 0057 — Rule-credit Double DQN Base / ALL

Date: 2026-11-01
Status: User-authorized isolated implementation and launch after acceptance.

This experiment inherits the complete physical, public observation and rule
projection contract of ADR0056, cloud source 4ee989c, tree
e1d038ed67a939f02540bf798bcb1cf9fbe03d77. It changes the learning mechanism to
fresh masked Double DQN, not the physical problem or common reward. No PPO
weights, old peer-pressure DQN states or old result files initialize these runs.

One seed4101, native Base and native ALL attention, each1000 complete environment
episodes. A completed environment episode may be naturally completed or physically
truncated at last release+2000; truncation remains visible and counts toward1000.
World indices0..999 are the cloud list, with normal/low/medium/high fault conditions
repeated250times. These labels do not denote arrival pressure. The raw common
reward and division by1000 remain unchanged; no individual rewards are introduced.

The candidate encoder including rule identity bits45:48 and the native ALL
candidate-linked attention are initialized afresh and trained as scalar Q
functions, never as a PPO loss or softmax probability. Semantic rule identities
and resolved physical actions bind replay choices. Alias representatives,
feasibility, public features and social links remain canonical. Automatic waiting
with no requested decision is not a new action. All forced legal decisions enter
Q learning. Native Base has no social extension; ALL is not capacity matched.

Fixed hyperparameters: Adam3e-4, Huber loss, gradient norm clip1, gamma1, epsilon
0.1 during training, replay10000transitions per role, three sampled batches of
at most256 per nonempty role after each episode, target copy every100global
optimizer updates. Each role shares parameters across its physical owners and
stages. Same-owner successors may cross stages; different owners never connect.
Reward is the common physical reward sum from current decision to that owner's
next decision. Double DQN selects the next legal action with online Q and values
it with target Q. Gamma1 applies also to same-tick decisions.

Natural-terminal tails include all remaining reward with zero bootstrap.
At physical truncation, unresolved final transitions per owner are omitted and
counted; earlier transitions remain. This differs from PPO critic bootstrap and
is disclosed, not called a critic-only intervention. No fabricated next-world
state or privileged critic value enters DQN targets.

Checkpoints preserve online/target weights, Adam, lossless compressed replay,
random generators, counters, and authoritative completed-episode transaction.
An uncommitted interrupted episode can be redone from its saved deterministic
episode boundary; this is not simulator-midstep continuation. Source, world,
configuration and protocol identities reject mismatched recovery. Initial,
500episode and1000episode checkpoints are prespecified, never selected by score.

Evaluation uses the cloud four fixed worlds/action seeds at each prespecified
checkpoint: greedy-Q primary, epsilon-greedy0.1 diagnostic. The midpoint and
initial-greedy panels are additional descriptive checkpoints relative to the
cloud's small original panel. Epsilon-greedy is not PPO sampling. Evaluation
does not update weights and reports all failures, incomplete orders, lower bounds,
quality coverage and common return; full mean tardiness is NA when censored.
Only one world per fault condition and one training seed limit inference.

The user removed wall-clock training limits for these two runs only. No physical
horizon or memory protection is relaxed. Two independent nice10 single-thread
workers use a fresh directory; existing four-worker DQN coordinator, its budget,
source and results remain untouched. RSS8GiB, host available8GiB and free disk2GiB
guards fail with retained receipts. No blind restarts or deadline-based case cuts.
Formal admission requires clean tagged source, native tests, independent reviews,
full episode engineering probes and full-capacity ALL replay/update/checkpoint
resource acceptance. Engineering outputs are never counted as scientific training.
