# Batch 02 · Diagnose Base

**Current development design (status checked 2026-10-02):** Three V4 Experiment files in
[`configs/runs/batch02/`](../../configs/runs/batch02/) form one native directory
batch. `smartsom check configs/runs/batch02` passed for all three files and 10
expanded entries. After an earlier calibration failure, the fixed-layout,
calibration-off revision ran both rule controls and six PPO entries before the
parent was stopped. Two separate joint-shape supplements subsequently finished
training; their held-out cases were only partially completed. See the
[result record](batch-02-results.md) for attempt-specific evidence. The command is
`smartsom batch-run configs/runs/batch02`. A parent run directory under
`runs/base-batch-02-v4-native/` freezes the source, file list, individual inputs,
stage status, calibration, results and progress. [ADR 0032](../decisions/0032-v4-directory-batch-execution.md)
defines the execution contract. Formal experimental status still requires an
integrated `main` commit or explicit tag; a dirty development run is labeled
accordingly.

| Stage | Experiment files | Variation and purpose |
| --- | --- | --- |
| 10 · rules | `control_auto.yaml`, `control_zero.yaml` | Same five-case Small H1/Low pool, auto versus zero travel. The auto control must have five engineering-clean cases, a first pickup and at least one qualified delivery in every case; zero may deliver zero but must complete cleanly. |
| 20 · Base | `train_matrix.yaml` | Four PPO Algorithms × seeds 101/1101 = eight independent `train-evaluate` entries, all with auto travel. |

The learning stage crosses unchanged reward versus Dispatcher missed-pickup
penalty 0.05 and rule Machine/Buffer versus jointly learned Machine/Buffer.
Every entry trains for 16,384 physical ticks, uses 4096-tick cases, validates
on five frozen cases every four updates, selects its own best **whole policy
composition** by complete cases → mean qualified deliveries → mean raw return
→ earliest update, and evaluates that best on five held-out cases. Data seed
909 pairs the physical cases across Algorithms; training seeds are 101 and
1101. Two seeds diagnose a mechanism; they do not establish a population-level
effect. Report each seed, paired delivery differences, zero-delivery cases,
missed-pickup boundaries, congestion and utilization. Do not substitute a
global best across algorithms or an unvalidated `last` checkpoint.

The directory runner first smokes **all 10 expanded entries** for up to 128
ticks each, without updating learner weights. This revision explicitly sets
`calibration_level: off`, `mode: performance`, `scheduling: fixed`, one
environment, zero sampling processes, one numerical thread and at most two
concurrent PPO entries. It runs **no performance probes** and does not read a
historical profile. The previous failed development calibration measured valid
two-entry candidates for all four learner groups at roughly 28–35 aggregate
physical ticks/s; that is a rationale for this fixed ceiling, not a guarantee
of formal-run speed. Because the new run is uncalibrated, admission begins with
one trial and grows only after committed updates and observed resource peaks.
All short engineering smokes still run. These controls change execution layout,
not reward, workloads or frozen formal cases.

The eight PPO entries form one ordered Tune queue. Finished entries release
slots for waiting entries, subject to the stage's maximum declared concurrency,
per-file and per-group limits, and live CPU/RAM admission. Without compatible
mixed-probe evidence, cross-group refill is marked schedule-uncalibrated; a
pair measured faster separately does not overlap. Stage 10's rule gate still
precedes any PPO training. [ADR 0033](../decisions/0033-v4-performance-profiles-and-continuous-queue.md)
defines the new execution boundary.
The earlier one-hour-per-probe Experiment drafts are retired. The batch has no
overall hard deadline. The stopped parent and separate supplements do not establish a single
end-to-end batch runtime; training and held-out evaluation may take hours.
Stop/resume acts on the parent run and reuses its saved
calibration rather than starting another performance test. If a Tune trial
fails, `smartsom resume PARENT --retry-failed` retries failed entries while
retaining completed results. Resume requires the frozen source and dependency
identity to remain compatible. The two files under `Resume/` are fresh
`train-evaluate` supplements with separate output directories, not checkpoint
continuations of the stopped parent.

The [result page](batch-02-results.md) separates the V4 parent and supplement evidence from the
historical r4 pilot. The r4 custom launcher completed controls and eight PPO
train-only entries but its eight attempted checkpoint evaluations had 0/0
frozen cases. Its saved files remain evidence of that
failure, not results of this new directory batch. Earlier 600-second resource
probing also exhausted its time on five long validation cases. The current revision skips performance probes while retaining short engineering
smokes and the same declared cases.
