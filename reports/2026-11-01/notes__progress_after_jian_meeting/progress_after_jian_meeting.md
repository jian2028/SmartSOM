# Progress log — plan after the Jian meeting

Plan: `notes/plans/2026-10-29_plan_after_jian_meeting.md`. Local only; no commit/push/messages.

## 2026-10-29

### Phase A — integration
- New merged copy `/Users/deng/code/SmartSOM-integrate3` = origin/main c6aa68e (#27, #29, #30)
  + Justin's `origin/codex/outage-aware-agv-controller` (staged merge, not committed)
  + our local changes (from `/Users/deng/code/SmartSOM-integrate2`).
- Core files (`engine/production.py`, `engine/production_protocol.py`, `_copy.py`): kept main's
  version; Justin's edits there were performance-only and main already has equivalents.
- From Justin: AGV planner/dispatcher, rule changes, tests, configs, validation docs.
- Conflicts: `production_rules.py`, `rule_registry.py` (union of rule names; removed a
  duplicated `validate_builtin_parameters`), CI test registry (union, de-duplicated).
- Backup of the previous copy: `/Users/deng/code/SmartSOM-backup-20261029`.
- Lint/format pass; 173 targeted tests pass; full unit suite running.
- Full unit suite on integrate3: 3817 passed, 2 failed (the known digest-pinned test; one test
  needs `sb3_contrib`, which this environment lacks). Neither is a code regression.

### Phase B — metrics
- `src/smartsom/experiments/evaluation_metrics.py` (integrate3): per-episode machine utilisation
  (processing / duration), AGV utilisation, delivery ticks, outage events → `result.json`
  `evaluation_metrics`. Analysis script: `/Users/deng/code/SmartSOM-tools/v2_metrics.py`
  (summary + `--paired` adaptability/robustness).
- AGV activity fix: the dispatcher always gives empty AGVs a target, so "has target" read 0.97
  even under rules. New definition: active = loaded, plus empty driving/service ticks that end
  in a pickup on the same target; waiting, parking or abandoned trips count as idle.
  Rule evals restarted with this (old partial runs moved to `runs/v2/_stale`).
- First partial numbers (5 episodes, old AGV definition): Justin rules low: thr 25.7/1000 ticks,
  makespan 2495, mean tardiness 41.5, late jobs 10.8, machine util 0.546; twin gaps
  m2/m7 0.27, m3/m6 0.30 → twin imbalance exists even under rules.

### Phase D — joint decision design
- `notes/plans/joint_decision_design.md`: literature summary + options. Recommend Option 1
  (empty AGV picks a (source, destination machine) pair; destination locked; same-tick
  exclusive) as the first conflict-free job→machine→AGV implementation in the policy layer.

### 2026-10-30 00:00 UTC — power loss
- The laptop lost power; all rule evals stopped after 1 episode each (partial runs → `runs/v2/_stale`).
- New AGV definition checked on those episodes: Justin rules AGV util ≈ 0.46–0.48 (was 0.97).
- Restarted with `/Users/deng/code/SmartSOM-tools/chain_v2.sh` (nohup): 6 rule evals, then
  C1 training (`configs/runs/central01/v2/c1_train.yaml`). Done lines go to `v2_logs/rules_done.txt`.
- Decision (Morris): C1 keeps passing_weight 0.1 (as R2), so the policy still pays for defects
  when it chooses machine modes (slow/normal/fast). "Passing" = quality pass rate, not shaping.

### 2026-10-30 00:40 UTC — partial rule results (9/20 episodes Justin, 2–3/20 random)
| Run | thr/1000 | makespan | mean tard | late | mach util | AGV util | twin gaps 1/8,2/7,3/6,4/5 |
|---|---|---|---|---|---|---|---|
| Justin low | 25.8 | 2481 | 45 | 11.7 | 0.550 | 0.457 | .07 .30 .30 .07 |
| Justin balanced | 26.6 | 2411 | 28 | 7.1 | 0.564 | 0.476 | .10 .34 .36 .06 |
| Justin low + m3 down | 25.3 | 2536 | 76 | 16.0 | 0.539 | 0.454 | |
| Justin balanced + m3 down | 25.7 | 2493 | 47 | 11.7 | 0.547 | 0.465 | |
| Random low / balanced | 1.2 / 2.4 | none finish | (delivered only) | | 0.08 / 0.12 | | |
Paired m3 outage [1200,1500): adaptability (deliveries during outage vs no outage) 0.69 low / 0.73
balanced, only 1/9 episodes ≥ 0.95; recovery to 95% rolling throughput 719 / 478 ticks after repair.
- Random-rule episodes hit the 4096-tick limit (~10 min each) → ~3 h; C1 training started now in
  parallel instead of waiting (chain stopped; rule evals continue).

### 2026-10-30 01:50 UTC — Justin rule baselines final (20 episodes each, clearance mover)
| Run | done | thr/1000 | makespan | mean tard | late | mach util | AGV util | twin gaps 1/8,2/7,3/6,4/5 |
|---|---|---|---|---|---|---|---|---|
| Justin low | 19/20 | 24.8 | 2542 | 61 | 12.5 | 0.532 | 0.447 | .07 .32 .27 .07 |
| Justin balanced | 19/20 | 25.7 | 2446 | 30 | 9.0 | 0.548 | 0.470 | .10 .32 .32 .06 |
| Justin low + m3 down | 20/20 | 24.1 | 2675 | 89 | 16.9 | 0.518 | 0.449 | |
| Justin balanced + m3 down | 19/20 | 24.8 | 2538 | 52 | 13.4 | 0.532 | 0.463 | |
Paired m3 outage [1200,1500), 20 pairs: adaptability 0.65 low / 0.70 balanced (1/20 ≥ 0.95);
recovery 669 / 459 ticks after repair (none censored); makespan +144 / +89; mean tardiness +29 / +22.
- One episode in 20 does not finish 64 jobs within 4096 ticks even under Justin rules.
- C1 training running (`runs/v2/20261030T005724Z-c1_train-a0bdce4b1c`); ~78 updates of 1024 after
  ~50 min → roughly 10 h total.

### 2026-10-30 04:15 UTC — random rule baselines final (20 episodes, clearance mover)
| Run | done | thr/1000 | mean tard (delivered only) | late | mach util | AGV util |
|---|---|---|---|---|---|---|
| Random low | 0/20 | 2.4 (≈10 jobs) | 1128 | 8.9 | 0.128 | 0.84 |
| Random balanced | 0/20 | 3.1 (≈13 jobs) | 1465 | 11.6 | 0.131 | 0.85 |
- Caveat: AGV util counts loaded time even when the AGV is stuck carrying a job it cannot drop,
  so random looks "busy" (0.84). Consider reporting loaded-moving vs loaded-waiting separately.
- C1 speed check: earlier long runs took ~9.7–9.9 h (R2-long, R5-long); C1 is on track for ~10.7 h.

### 2026-10-30 09:10 UTC
- C1 at ~900–940/1024 updates; expected to finish ~10:15 UTC.
- Added `agv_loaded_waiting_mean` to evaluation metrics (loaded AGV not moving, not servicing).
- Auto-launcher queued: `/Users/deng/code/SmartSOM-tools/c1_evals.sh` starts once all 5 seeds reach
  update-001024: per model, balanced {softmax, argmax} and balanced + m3 outage {softmax, argmax},
  20 episodes each, eval seed 101 (same cases as the rule baselines). Configs →
  `configs/runs/central01/v2/c1eval/`, done lines → `v2_logs/rules_done.txt`.

### 2026-10-30 15:55 UTC — C1 trained; built-in softmax eval (100 episodes, training seeds)
- C1 eval configs failed at first (relative paths one level short); fixed, all 20 relaunched ~15:50.
- Built-in eval: completed 53/100 (per seed 6, 10, 15, 11, 11 of 20); thr 17.7/1000 (Justin 25.7);
  makespan of completed episodes 2560 (Justin 2446); mean tard 32; late 11.9; machine util 0.37;
  AGV util 0.47, loaded-waiting 0.10; twin gaps 1/8 .38, 2/7 .18, 3/6 .29, 4/5 .31.
- So C1 is clearly below Justin rules and only half the episodes finish → likely stalls/deadlocks
  with the clearance mover; check after the controlled evals.

### 2026-10-30 22:20 UTC — C1 controlled evals final (5 models × 20 episodes, eval seed 101 = rule cases)
| Policy | done | thr/1000 | makespan* | mean tard | late | mach util | AGV util | loaded-wait | twin gaps 1/8,2/7,3/6,4/5 |
|---|---|---|---|---|---|---|---|---|---|
| Justin rules | 19/20 | 25.7 | 2446 | 30 | 9.0 | 0.548 | 0.470 | – | .10 .32 .32 .06 |
| C1 softmax | 53/100 | 17.7 | 2545 | 32 | 11.9 | 0.371 | 0.469 | 0.101 | .38 .18 .29 .31 |
| C1 argmax | 68/100 | 21.8 | 2389 | 24 | 6.4 | 0.358 | 0.436 | 0.086 | .57 .38 .55 .39 |
| C1 softmax + m3 down | 40/100 | 15.5 | 2618 | 50 | 13.5 | 0.335 | 0.448 | 0.109 | |
| C1 argmax + m3 down | 62/100 | 20.8 | 2412 | 27 | 9.8 | 0.346 | 0.440 | 0.093 | |
| Random rules | 0/20 | 3.1 | – | – | 11.6 | 0.131 | 0.85 | – | |
*makespan over completed episodes only. Per-model completed (argmax): 8, 9, 17, 18, 16 of 20.
Paired m3 outage (vs same policy without outage): argmax adaptability 0.69 (9% ≥ 0.95), recovery
417 ticks, 13/100 never recover; softmax 0.78 (20%), 475 ticks, 28/100 never. Justin: 0.70, 459, 0.
- Processing ticks per delivered job: Justin 171, C1 argmax ~123–131 → C1 mostly picks fast mode
  (0.8× time, higher defect rate). That is why C1 machine util is low; util is not comparable across
  policies with different modes unless modes are fixed or util counts "busy incl. mode".
- When C1 (argmax) finishes, it beats Justin on makespan (2389 vs 2446), tardiness and late jobs;
  the problem is the ~32% of episodes that stall (loaded AGVs waiting ~9% of time).
- Twin imbalance is worse under C1 (gaps .4–.57).

### 2026-10-31 — C1 stall diagnosis
- Recorded replay of a stalled argmax case (model seed 104, case 1: 61/64 jobs, last delivery 2264)
  reproduced it exactly. At the stall: all 8 AGVs empty, all machines idle, 3 jobs waiting in
  machine output buffers (buffer_006 = m2 post ×2, buffer_016 = m7 post ×1), all movers WAIT forever.
  5 AGVs target buffer_001/port_007 (the release source, exhausted after all 64 released) and jam
  the single-lane row y=0 head-on with the 2 AGVs heading for port_004 (where the jobs are).
- Cause: when no source has supply, our exclusive mask falls back to every legal target, and the
  policy learned to send idle AGVs to the release source (useful during arrivals, harmful after).
  Neither protocol offers an "idle" dispatch target. Justin's dispatcher admits ≤1 empty AGV per
  port and parks the rest at the nearest target.
- Fix (opt-in `dispatch_idle_fallback`, needs `dispatch_tick_exclusive`, policy layer only):
  ≤1 empty AGV heading for any port; with nothing to fetch, park at the nearest port nobody else
  targets. Tests: `tests/unit/test_dispatch_policy_options.py` (+2, 16 pass).
- Testing it on the trained C1 models without retraining (copies with the flag in model.json:
  `runs/v2/models_idlefix/seed10x`), first on the stalled case, then 5×20 argmax balanced.
- First fix alone: stalled case still stalls (59/64). Model copies needed their
  `effective_model_sha256` recomputed after adding the flag (self-consistency hash; weights unchanged).
- Second deadlock (recorded): 4 loaded AGVs all committed to m6's single input port (3,8) while
  twin m3 idle; plus parked empty AGVs in the bottom corridor → mover gridlock.
- Extended `dispatch_idle_fallback`: also ≤1 loaded AGV heading for any port (spreads loaded AGVs
  over twin machines). Re-probing.
- Loaded one-per-port rule made it worse (probe: case 0 57/64, case 1 63/64 — both stall). Both
  stalls sit in the dead-end bottom corridor around m6 (ports at (3,8), (5,8)): too many AGVs enter
  and the clearance mover cannot untangle them. Reverted the loaded rule; kept the empty-AGV rule.
- Launched 5×20 argmax balanced with the empty-AGV fix (`configs/runs/central01/v2/c1fix/`) to
  measure the completion rate vs 68/100.

### C1 + dispatch_idle_fallback (eval-time only, no retrain) — 2026-10-31
Argmax, balanced, 5×20, seed 101. Completed per seed (fix vs original): 101 18/16, 102 10/8, 103 18/18, 104 8/9, 105 14/17 → **68/100 vs 68/100**.
Throughput 22.2 vs 21.8; machine util 0.364 vs 0.358; loaded-waiting 0.083 vs 0.086. Seed-level tardiness mixed.
Conclusion: the empty-AGV fix moves the stalls around but does not reduce them; the remaining deadlocks are loaded-AGV gridlocks (e.g. m6 single input port, bottom corridor). An eval-time mask is not enough → needs retraining with the rule and/or a mover-level deadlock fix.

### C1 with Jian's `shortest_path_yield` mover (eval-time swap, no retrain) — 2026-10-31
Argmax, balanced, 5×20, seed 101. Completed per seed: 101 19, 102 16, 103 17, 104 16, 105 19 → **87/100** (orig 68/100; Justin rules 19/20).
Throughput 23.2 (orig 21.8, Justin 25.7); makespan 2633 (orig 2389, completed eps only); tardiness 99 (orig 24); late jobs 24.4 (orig 6.4); machine util 0.377; AGV util 0.66 (side-steps count as driving); loaded-waiting 0.141.
Conclusion: step-aside breaks most gridlocks (+19 completed episodes) but slows transport → tardiness ×4. The model was trained with clearance_shortest_path, so this is a mismatch; next candidate = retrain C1 with the yield mover.

### Justin dispatcher + Jian's yield mover (fair-comparison baseline) — 2026-10-31
Balanced 20 eps: completed 19/20; throughput 24.3 (clearance 25.7); makespan 2591 (2446); tardiness 46 (30); late 13.5 (9.0); machine util 0.522; AGV util 0.61.
m3 outage: completed 19/20; throughput 23.8; makespan 2665; tardiness 67. Paired: adaptability 0.735 (pass 3/20), recovery 462 ticks (1 censored), makespan +76.
Yield costs Justin's rules ~5% throughput and +16 tardiness, vs C1 where it cost +75 tardiness (model not trained with it). C1y retraining in progress (started 02:21 PDT).

### C1y = C1 retrained with yield mover — training done 12:41 PDT 2026-10-31 (~10.3 h)
Built-in final eval (softmax, balanced, 5×20): completed **92/100** (C1 53/100); throughput 23.5 (17.7); makespan 2658; tardiness 95 (32); late 27.8 (11.9); machine util 0.475 (0.371); AGV util 0.73; twin gaps .24/.14/.33/.24 (more balanced).
Argmax balanced, argmax bdm3, softmax bdm3 evals (15 runs) launched 12:5x PDT.
C1y evals (5×20, eval seed 101), finished ~17:30 PDT:
| policy | completed | thr | makespan | tard | late | mach util | AGV util |
|---|---|---|---|---|---|---|---|
| argmax balanced | 89/100 | 21.5 | 2929 | 227 | 33.9 | 0.419 | 0.678 |
| argmax bdm3 | 79/100 | 20.7 | 2956 | 248 | 35.6 | 0.406 | 0.673 |
| softmax balanced (train final eval) | 92/100 | 23.5 | 2658 | 95 | 27.8 | 0.475 | 0.728 |
| softmax bdm3 | 92/100 | 22.7 | 2758 | 126 | 31.5 | 0.463 | 0.721 |
Seed 104 is an outlier under argmax (tard ~587, thr 17.4); other 4 seeds argmax tard ~138.
Argmax paired adaptability per seed: .98/1.10/.74/.82/.76 (mean 0.88; pass ≥0.95 ≈ 32%); recovery 584/485/898/512/431 (mean 582). Justin+yield: 0.735, 15%, 462.
Softmax-bdm3 not paired (no softmax balanced run at eval seed 101 — train final eval may use different seeds).
Takeaway: softmax > argmax for C1y; throughput near Justin+yield (−4..7%), better adaptability, but tardiness 2–3× Justin. Next: tardiness.

### Tardiness diagnosis (recorded probes, 2 eps each, balanced) — 2026-10-31 night
Tool: SmartSOM-tools/tard_breakdown.py (ticks per job by location, release→ship).
| run | tardy/126 | lateness of tardy | tardy jobs: agv / pre-queue / machine / post-wait / inspection / rework | mode mix |
|---|---|---|---|---|
| C1y softmax s101 | 27 | 142 | 175 / 145 / 207 / 84 / 102 / 91 | slow .41 normal .31 fast .28 |
| C1y argmax s104 | 73 | 669 | 192 / 261 / 221 / **457** / 20 / 0 | **slow .99** |
| Justin+yield | 16 | 171 | 144 / 199 / 236 / 76 / 55 / 306 | normal 1.0 |
Findings: (1) argmax s104 is a degenerate policy: all-slow mode and finished jobs wait ~457 ticks for pickup → argmax outliers are a collapse, not a tardiness-trade-off. (2) Softmax: lateness per tardy job similar to Justin; gap = more tardy jobs (21% vs 13%) from slightly longer transport and inspection waits; rework is NOT the cause (lower than Justin). Small sample.

### C1yt4 = C1y with tardiness_weight 4.0 — training done 08:03 PST 2026-11-01
Softmax balanced final eval 5×20: completed 91/100 (C1y 92); thr 22.9 (23.5); makespan 2723 (2658); **tardiness 103 (95)**; late 27.2 (27.8); mach util 0.467.
→ Doubling the tardiness weight did NOT reduce tardiness. 15 evals (argmax bal/bdm3, softmax bdm3) launched ~08:10.

## 2026-11-01 Phase E (joint job→machine→AGV) implementation check
- Code lives in `~/code/SmartSOM-phaseE` only (default-off flag `network.dispatch_joint_destination`); integrate3 is untouched while runs are live.
- Unit: new joint tests pass (19/19 in test_dispatch_policy_options). Full suite: 5 failures, all environmental/flaky and reproduced in integrate3 without Phase E (missing sb3/wandb modules, process-timing tests under load); plus the pinned legacy digest test, which any production_models edit changes (expected).
- E1 smoke (seed 101, 8192 ticks, 32 updates): no errors, explained variance 0.82, entropy 1.63. Loaded AGVs never switched delivery target (7865 consecutive delivery steps, 0 switches).
- E1 smoke final eval (2 eps, after only 8192 training ticks): 2/2 completed, makespan 3045, throughput 21.1/1000, tardiness 15.5k (untrained-level; not a result).

## 2026-11-01 Pre-training changes (all in ~/code/SmartSOM-phaseE, default-off)
Decision (Morris): keep C1y continued training running; finish and verify all code changes before choosing the next training batch. C1yt4 bdm3/argmax evals stopped (softmax already showed no gain). Stopping left 30 orphaned spawn workers burning CPU (load 139); killed.
1. Phase G metric (`SmartSOM-tools/v2_metrics.py --paired`): every outage scored separately (breakdown/repair paired per machine); recovery searched until the next breakdown, else censored. On existing scripted m3 data: Justin+yield 0.714 / 18% pass / 439 ticks (old single-outage code: 0.735 / 15% / 462; the difference is the second m3 outage at 2700 now counted).
2. Random breakdown factory `configs/factories/batch02_small_h1_bd_random.yaml`: machine_001-004 (one per twin pair) fail, exponential uptime mean 3000, repair 100-300. Sampled: 3.6 outages per 3000-tick episode, 2% episodes with none, ~690 machine-down ticks.
   Simplification vs plan: plan said "at most one machine of a pair down at a time"; this uses "only one machine of each pair can fail".
3. `training.resample_outages` (TrainingV4/TrainingOptionsV3, composable TrainingSession.new_sim): fresh reliability schedule per training episode from the episode seed. Before this, every training episode reused the one frozen schedule. Verified in e1g_smoke: episodes drew 6/4/8 different outages; eval cases differ too.
4. Phase F option (a) (Morris chose a over b): `task.starvation_weight` (training-only): per tick subtract w * starved/(n_machines * reference_ticks); starved = IDLE, not down, own input buffer empty, and some picked-up job (not queue/system input/inside a machine/FAIL) elsewhere needs its operation next. Rejected "busy fraction" reward because slow mode inflates processing time.
5. Bug fix: evaluation_metrics outage events read `machine_id` but the engine emits `machine`, so outage machine was always null (also in integrate3; harmless for single-machine scripted tests).
6. Phase E real-sim probe (E1 smoke model, 3 eval eps): when a commitment is enforced the AGV always delivers to the chosen machine (565/565), but 269 loaded requests (32%) fell back; and 2997 of 3909 empty-AGV choices had no pair. Reason breakdown pending.
Tests: joint 19/19, reliability 27, reward 86, config 264 pass.
7. Phase E fallback fix. Probe on the E1 smoke model (1 eval ep each):
   - original: 125/313 loaded deliveries fell back (40%), 121 because the vehicle received a job for another operation.
   - pairs follow the job the vehicle would receive in ready order: 61/214 (28%), all wrong-operation.
   - root cause: the source's release order is the controller's own buffer decision. Added `release_mask`: a source releases jobs for the operation of the machine chosen by a committed empty vehicle waiting at its port (prefix-aware). Result: 11/185 (6%), all commits otherwise delivered to the chosen machine.
   Remaining no-pair empty choices are mostly sources with no ready job yet (pre-positioning), by design.

### 2026-11-01 E1 training launched
- Morris OK'd E1 (C1y + dispatch_joint_destination with release_mask), 5 seeds 101–105, 262144 ticks, same budget as C1y.
- Config: SmartSOM-phaseE `configs/runs/central01/v2/e1_train.yaml`; run dir `runs/phaseE/20261101T223155Z-e1_train-507549a762` (started 14:31 PST, after ~17 min prepare).
- Verified: 5 entries training (update 13 at 14:43), no tracebacks. Watcher: `watch_runs_root.sh e1_train 5` (ROOT=runs/phaseE).
- Next: when C1y continued training finishes tonight, launch E1a (starvation w=1.0) and E1g (bd_random + resample_outages).

- 2026-11-01 21:40 Morris: stop C1y continued training (stopped, no orphans). W&B bridge for E1 started (bridges/e1_train.log). E1a launched 22:18 PST: runs/phaseE/20261102T061812Z-e1a_train-980e63f087, training confirmed, W&B bridge bridges/e1a_train.log. E1g (bd_random + resample) to launch after E1 finishes (Morris: separate launches).
