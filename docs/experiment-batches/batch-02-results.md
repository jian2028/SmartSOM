# Batch 02 · Results

**New V4 directory batch:** Three Experiment files and 10 expanded entries pass
input checking. The 2026-09-29 [development launch](/Users/jianni/code/SmartSOM/runs/base-batch-02-v4-native/20260929T103956Z-batch-batch02-26430b83ac/run.json)
passed all 10 short smokes and measured valid calibration candidates for all
four performance groups. At the five-minute deadline, a final repeat probe
with less than a second remaining was misclassified as an engineering error;
the parent run failed before either rule control, PPO training or held-out
evaluation began. Its [calibration report](/Users/jianni/code/SmartSOM/runs/base-batch-02-v4-native/20260929T103956Z-batch-batch02-26430b83ac/performance/20260929T103957Z-tune-01554e7d07/calibration.json)
and logs are diagnostic evidence, not a Base result.

**Calibration-off development run (checked 2026-10-02):** The subsequent
[parent ledger](../../runs/base-batch-02-v4-native/20260929T114137Z-batch-batch02-16120e56df/batch.json)
records skipped calibration, completed engineering smokes, both rule controls
completed with five-case gates passed, and a stopped parent. Its
[Tune ledger](../../runs/base-batch-02-v4-native/20260929T114137Z-batch-batch02-16120e56df/performance/20260929T114138Z-tune-60a6cdda95/batch.json)
records entries 1–6 completed at 16,384 physical ticks / 64 updates, entry 7
failed and entry 8 queued. The child's stale `running` snapshot does not mean
that a process is still running; the stopped parent is authoritative.

**Separate joint-shape supplements:** The two configurations in
[`Resume/`](../../configs/runs/batch02/Resume/) start fresh `train-evaluate`
entries rather than resume the parent's checkpoints. Their
[parent ledger](../../runs/base-batch-02-v4-supplement/20260930T093656Z-batch-Resume-ded7aff25e/batch.json)
and [Tune ledger](../../runs/base-batch-02-v4-supplement/20260930T093656Z-batch-Resume-ded7aff25e/performance/20260930T093657Z-tune-2d852eddf9/batch.json)
record both entries completed at 16,384 ticks / 64 updates. Training completion
is separate from qualified-demand completion in the held-out evaluation:

| Seed | Training ticks / updates | Complete cases | Truncated cases | Exceptions | Mean qualified deliveries | Evaluation evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 101 | 16,384 / 64 | 1/5 | 4/5 | 0 | 62.4 | [Summary](../../runs/base-batch-02-v4-supplement/20260930T093656Z-batch-Resume-ded7aff25e/performance/20260930T093657Z-tune-2d852eddf9/experiments/joint_shape_seed101__entry-0001/attempt-0001/evaluation/summary.json) |
| 1101 | 16,384 / 64 | 0/5 | 5/5 | 0 | 12.4 | [Summary](../../runs/base-batch-02-v4-supplement/20260930T093656Z-batch-Resume-ded7aff25e/performance/20260930T093657Z-tune-2d852eddf9/experiments/joint_shape_seed1101__entry-0001/attempt-0001/evaluation/summary.json) |

These attempts used dirty development source. They do not establish a formal
integrated-main experiment, an all-cases acceptance, or a causal improvement
from shaping. Keep each attempt's source identity and original evidence intact;
committing the current configuration does not retrospectively change that status.
The linked `runs/` records are local evidence, intentionally excluded from Git.
Do not mix these attempts with the historical r4 pilot below.

**Historical r4 status:** r4 finished with engineering evaluation failure. Both resource probes, both rule controls, and all eight PPO training runs completed. The launcher selected a best checkpoint for each seed, but every subsequent CLI evaluation ran **0/0 cases** because train-only V4 runs froze no evaluation inputs; all eight were marked `incomplete_cases`. No held-out Base result was measured by that r4 attempt. The original failed evaluation files remain as evidence. A separate manual recovery launcher was prepared and checked, but **not started**; its source is retained in local `artifacts/batch02-retired-20260929/scripts/`, outside the active interface.

The r4 [bundle manifest](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4-r4/manifest.json) freezes the source, inputs and launcher. After execution, [status](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4-r4/status.json), [checkpoint selection](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4-r4/selection.json), [selected evaluations](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4-r4/evaluations.json), and local logs are the evidence. Missing files mean the stage has not run; they are not zero outcomes.

| Arm | Seed | Training | Selected update | Five held-out qualified deliveries | Complete cases | Zero cases | Missed pickup boundaries | Evidence |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |

The analysis will compare paired seeds, diagnose pickup and destination waiting, and state all partial or failed stages.

The first bundle stopped after its 600-second resource probe measured no valid candidate. It completed no rule control, PPO training or selected evaluation; see [failed status](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4/status.json) and [calibration report](/Users/jianni/code/SmartSOM/runs/base-batch-02-auto-v4/experiments/20260928T200631Z-batch2_calibration_rule-94c8a555d5/performance/20260928T200632Z-tune-365df6dc06/calibration.json).

The archived manual recovery plan would read the same five frozen auto-control test worlds and each selected checkpoint, verify source and physical-contract identity, and write to `evaluations_manual-01` and `recovery-manual-01` without altering r4 evidence. Previous partial recovery attempts were interrupted on user request.

The V4 compiler and saved-run evaluation entry have also been corrected generally: new train-only V4 runs freeze held-out worlds for later checkpoint evaluation, while an older empty snapshot now raises an explicit error. This correction changes neither r4's frozen source nor its completed training evidence; its old checkpoints still require the manual recovery above.
