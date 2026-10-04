# Batch 03 — Native Windows execution protocol and provenance

Batch 03 indexes the October 3, 2026 Windows qualification, reproduction and
diagnostic campaign. It is a retrospective execution record, not one homogeneous
experiment planned in advance. Historical Batch 01 and Batch 02 remain separate
recipe/development records. [Results](batch-03-results.md) and the small
[evidence manifest](batch-03-manifest.json) describe what actually ran.

## Source boundary

The later experiments used the explicitly tagged source
`9f48eb775861d671abbc1a113de76a2caaa8e9fb`, from
[PR 5](https://github.com/jian2028/SmartSOM/pull/5), which was still unmerged when
this report was first prepared. PR 5 subsequently merged on October 4 as
`342c3bc76c029ee5f9f61c3ee95769131b82ed10`; this documentation is based on that
merge. The integration does not change the recorded experiment source identities
or constitute a rerun on `main`.
The earlier qualification used a dirty repair tree based on `b647637a`; the
initial four-hour pilot used clean `7ae74f3`. Do not substitute one phase's source
identity for another. The qualification's original diff hash was incomplete;
its provenance note records that limitation.

Experiments ran on native Windows 10 build 19045, i7-9700K, 32 GiB RAM, CPU,
Python 3.12.15, Torch 2.14.0+cpu and Ray 2.58.0. The main runs used one environment,
zero sampling subprocesses and one numerical thread, with process-local BLAS
thread limits. No WSL or CUDA substitution was used.

## Distinct execution phases

| Phase | Scope and contract | Recorded outcome |
| --- | --- | --- |
| Qualification | CLI, physics/control/display gates, real terminal, real updates, checkpoint and same-environment resume checks | Engineering evidence; dirty repair-source limitation retained |
| Initial four-hour pilot | Rules on T10/T11/T12; T10 PPO; source `7ae74f3` | Budget-limited partial; not pooled with later runs |
| Batch01-recipe anchor | T10/H1/low, joint PPO, seed 101; 16,384 planned ticks; 256/update; validation every 4 updates on 5 worlds | 14,336 ticks / 56 updates; best update 8; recovery update 56 |
| Dispatcher reference | Dispatcher-only PPO with rule partners; seed 101 | 16,384 ticks / 64 updates; best update 16 |
| Fresh scouts A–E | Seed 101; 4,096 ticks / 16 updates; validation at 8/16 on 2 fixed worlds | Five completed runs; F never admitted |
| Policy-mode/random controls | Two 64-job development worlds; Dispatcher treatment varies while other roles use rules | Transport diagnostics, not a matched random baseline for the whole joint system |
| 128-job routing calibration | T10 and T11 rule controls; bounded T10 stall fork | T10 failed original calibration; learning comparison blocked |
| Final evaluations | Three unique selected-model sets, each on the same 5 worlds | 15 unique closed cases; A reuses identical anchor evaluation |

The later learning recipe uses 64 finite demands (48 common, 16 novel), 30-tick
arrivals, low variation, 4,096-tick episode horizon, geometry-derived positive
travel times and ceil-rounded processing. Joint PPO learns Machine, Buffer and
Dispatcher; Mover remains `automatic_travel`. Networks are 64×64 tanh, with
learning rate 0.0003, batch size 64 and 4 epochs. No new shaping or Social Learning
extensions were introduced. The built-in encoder includes public global resource
state/topology; this is not a strict local-observation/no-peer SO-MARL baseline.

Scouts A/B/C/D/E used `(gamma, lambda)` of `(.99,.95)`, `(.99,.95)`,
`(.995,.95)`, `(.999,.95)`, `(.99,.99)` respectively. B trained Dispatcher only;
the others trained all three decision roles. Parameter variants started fresh,
not from a trained checkpoint. Best selection ranked completed cases, mean
deliveries, then mean raw return; ties kept the earlier checkpoint. Final test
results did not choose checkpoints. The once-held-out worlds have now been
inspected and are diagnostic reuse for future work.

## Evidence access

Raw inputs, logs, traces, models and recovery state remain Windows-local. The
manifest uses paths relative to `WINDOWS_EVIDENCE_ROOT`, a locally supplied
evidence directory, not repository-relative download links. The central local
entry is `experiment-batches/batch-03/README.md`; the original final report is
`diagnostics-12h-20261003-v1/campaign-report-zh.md`. Their byte hashes and selected
run/checkpoint identities allow an authorized local reader to locate and verify
the records without publishing machine-specific paths or large artifacts.

This documentation PR runs no experiments and makes no runtime changes.
