# Batch 03 — Native Windows results

The pipeline ran, but the final learned policies did not complete the 64-demand
worlds. These are one-training-seed development results, not convergence or
statistical superiority evidence. [Protocol](batch-03-plan.md) and
[manifest](batch-03-manifest.json) retain the different phases and source versions.

## Fresh scouts

All five scouts completed 4,096 physical ticks / 16 updates with seed 101.

| Scout | Update 8 validation deliveries/64 | Update 16 deliveries/64 | Best update | Training + validation minutes |
| --- | --- | --- | ---: | ---: |
| A — joint original gamma/lambda | [9, 6] | [0, 0] | 8 | 36.81 |
| B — Dispatcher only | [0, 2] | [0, 1] | 8 | 33.96 |
| C — joint gamma .995 | [0, 0] | [14, 11] | 16 | 36.25 |
| D — joint gamma .999 | [0, 0] | [0, 1] | 16 | 28.25 |
| E — joint lambda .99 | [0, 1] | [0, 0] | 8 | 24.10 |

The five runs recorded 30,488 finite backend loss events; the recorded numerical
checks found actual updated weights and unchanged frozen partners. This is
engineering evidence, not evidence of useful policy quality. F was not admitted
within the remaining budget and is not a failed or zero-valued result.

## Selected-model evaluation

| Model | Best update | Deliveries/64 in the five paired worlds | Mean deliveries | Mean raw return | Completed / truncated |
| --- | ---: | --- | ---: | ---: | --- |
| Joint anchor | 8 | [8, 11, 13, 3, 9] | 8.8 | -8939.496 | 0 / 5 |
| C, joint gamma .995 | 16 | [7, 12, 19, 10, 12] | 12.0 | -8265.734 | 0 / 5 |
| Original Dispatcher-only reference | 16 | [1, 2, 1, 0, 0] | 0.8 | -10645.848 | 0 / 5 |

These are 15 unique completed *evaluation executions*, all of which ended in
simulation truncation. They are not 15 completed production episodes. Missing
completion makespan is unavailable, not zero; statistics restricted to delivered
jobs have survivor bias. C minus anchor deliveries were `[-1, 1, 6, 7, 3]` across
these worlds, but this is only a single training seed and selected checkpoints.

A/update8 and anchor/update8 had identical effective models and evaluation
inputs. A's evaluation therefore aliases the anchor's five cases. Separate
training provenance is retained; the reuse is not five more cases or an
independent replication. The anchor stopped cooperatively at 14,336/16,384 ticks
(56/64 updates), preserving best8 and recovery56. Its final validation delivered
zero in all five worlds; best-checkpoint results do not demonstrate late stability.

### Verified action-selection semantics

The native joint evaluation path at `9f48eb7` constructs `ModelPolicy` with
`training=False` and `deterministic=True`; `choose` uses argmax in that state.
`evaluate_cases` does not apply the configured deterministic option. Inspection
of the retained final wrappers and numerical instrumentation found no override.
The joint final results above therefore used **greedy argmax neural actions**,
not categorical sampling. This does not imply deterministic factory physics or
matching: their seeded random mechanisms remain present. No rerun was used to
reach this source/harness audit conclusion.

The separate Dispatcher-mode diagnostic explicitly set `deterministic=False`
and seeded the policy generator. Learned categorical Dispatcher and uniform
legal random Dispatcher both completed four diagnostic cases (two development
worlds × two sampling seeds). They used rule partners; completion alone did not
establish learned advantage. Sampling seeds are not independent training seeds.

## Calibration failures and interruptions

- T10's 128-job rule control reached a wall-time cap at 37/128 deliveries,
  tick 6,760. Loaded AGVs targeted full machine pre-buffers, with blocked post
  buffers. T11 completed 128/128 at makespan 4,282. The required original paired
  calibration failed, so no 128-job learning comparison was admitted.
- A 512-tick fork from the exactly replayed T10 prefix, with one legal redirect,
  added one delivery before the stall re-formed. This was a bounded diagnostic,
  not a general routing repair or proof of all physical correctness.
- An execution disconnect around 20:22 UTC interrupted the original Dispatcher
  final run after three closed cases and a partial fourth. Cases 3/4 were rerun
  from their original frozen starts. The first 1,278 trace entries matched the
  interrupted prefix byte-for-byte; this was not mid-episode continuation.
  The original partial trace and failed postprocessing records were preserved.
- Native Tune's durable commit path failed on file/directory fsync handling on
  Windows. Ordinary foreground experiments continued without removing durability
  barriers. A tiny two-environment/two-sampler recovery check passed at eight
  total ticks; it is not a throughput benchmark.

The later source came from [PR 5](https://github.com/jian2028/SmartSOM/pull/5),
which was unmerged during execution and merged on October 4 as `342c3bc`.
Earlier qualification and the initial pilot have distinct recorded source states.
Some historical raw statuses remain `running` after interruption; the final
reconciled ledger verifies their owners exited. No stale status was overwritten
to manufacture success.

## Local integrity audit

The retrospective index contains 105 run/case records: 59 non-case records and
46 case artifacts across nine phase categories. These are inventory counts,
not counts of independent experiments. Eleven best pointers and 17 inference
archives were checked, including weight/encoder or archive hashes. Indexed
references and 66 documentation links resolved locally. The audit does not claim
to rehash every large replay. Raw models, trajectories and recovery checkpoints
remain outside Git; no historical evidence was deleted or moved for this report.
