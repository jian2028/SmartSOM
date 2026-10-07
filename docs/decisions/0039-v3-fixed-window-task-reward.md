# 0039 — Explicit V3 fixed-window task reward

Date: 2026-10-07
Status: User-authorized objective; engineering verification separate.

Supersedes earlier base reward accounting only when the experiment explicitly
sets `training.reward.task`. Default recipes retain an explicitly identified
legacy-coefficient raw ledger; archived sources/results are never relabeled.
No global algorithm gamma/default, Batch04 file or episode cap changes here.

The task schema is smartsom.shipment-task-reward/v1. Defaults are shipment weight
0.5, passing weight 0.3, tardiness weight 0.2, reference jobs 128 and reference
ticks 4096. They belong to experiment reward configuration, never algorithm
parameters or observations. The task uses gamma=1; team/role reward transforms
cannot stack onto it. Positive learner scaling remains numerical scaling only.

Each physical interval emits the trainer scalar:
`0.5 * delta_S / 128 + 0.3 * delta_P - 0.2 * delta_TT / (128 * 4096)`.
S counts unique original shipments, good or bad. P is oracle good shipments/S,
computationally zero at S=0 but reported unavailable. Delta_P compares the whole
committed batch, not individual shipment event order. TT is an unweighted
integral of released, unshipped original-order lateness, including external FIFO
and replacements with original due dates. For [t,t+1], an order's increment is
max(0,end-due)-max(0,t-due), ending at its actual shipment timestamp or t+1.
Orders released at t+1 contribute from the next interval. No inspection,
conflict, WIP, revealed-FAIL or unfinished-cap term is stacked in task rewards.

The core owns private committed deltas/totals. The experiment learner replaces
the legacy scalar before collection; actor/critic/rule/public inputs receive no
quality deltas, task reward, references or oracle counts. Oracle scalar feedback
is an explicit training assumption; it does not imply observable quality at
execution. Public traces retain raw legacy-coefficient rewards; private task
reports identify their separate objective and source.

With gamma=1, the episode sum telescopes to 0.5*S/128 + 0.3*P -
0.2*TT/(128*4096). A task's scenario limit is a true finite objective horizon;
its last PPO packet has zero bootstrap, and pending DQN rewards commit to terminal
rows, even when shipping performance is incomplete and recorded status is
truncated. Training update cuts do not terminate the task. Default/legacy
truncation bootstrap and censoring remain unchanged. Task identity is frozen in
models/run records/continuations; restore rejects differing task contracts before
mutable state loads. Validation must explicitly maximize task return on all
valid objective windows, not a survivor-only makespan or completion ranking.

This is a soft three-objective tradeoff, not a strict throughput-first guarantee.
A fixed window bounds only observed overdue time; unfinished final tardiness and
makespan remain censored. Engineering tests do not establish better scheduling
or authorize a research claim. Formal Batch04 reruns remain a separate task.
