# 0094 — Use the official DQN checkpoint contract

Status: user-authorized; implementation under verification.

Supersedes the replay-persistence requirement and custom state extensions in
ADR0089 and ADR0093. The user accepts the Ray 2.60 new-stack checkpoint behavior:
restore the official model, target, optimizer and counters, but allow replay to
start empty and accumulate new experience. Do not promise uninterrupted-learning
equivalence, repeat the original warmup schedule, or restore partial simulator
state. Training follows official restored counters and scheduling.

Use DQNConfig and the official replay class without a SmartSOM DQN subclass,
replay inverse-index repair, counter repair or minimum-tree repair. Do not add
custom persistence or epsilon-greedy. Old custom checkpoints are not migration
inputs; their frozen source and evidence remain preserved.

The direct performance baseline must also use official DQN and transfer official
AlgorithmConfig state through Tune, rebuilding it with the public from_state
interface. This avoids the official dictionary replay-type conversion defect
without changing algorithm behavior or patching Ray.

Verification retains fresh-process shared/independent recovery, learner state
comparison, corrupted checkpoint rejection, continued learning and a second
restore. It explicitly checks empty replay after fresh restoration and new
experience accumulation. Best/last still reference official checkpoints. The
five-percent benchmark remains same-version, same-workload engineering evidence.
