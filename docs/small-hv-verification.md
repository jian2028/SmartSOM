# Small H/V 与运输矩阵：工程验收记录

2026-09-26；独立 worktree `/Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM`，
分支 `codex/composable-policies-v3`，基于 `517bf53dd2093a176896b371c25f0cd10cbc80cd`。
改动未提交；没有启动交付的 24 组训练。Social Learning 未改动。

## 最终检查

| 检查 | 新完成结果 |
|---|---|
| 完整 `pytest -x -q --durations=10` | **1849 passed、12 skipped、93 warnings，退出码 0**；1562.22 秒 |
| `ruff check .` | 通过 |
| `ruff format --check .` | 通过，最终检查 386 个 Python 文件 |
| `git diff --check` | 通过 |
| 最终 `study prepare`、`study show`、单组 `show-config`、重复 prepare | 通过；状态 prepared，未启动子任务 |

命令使用 `source .venv/bin/activate` 后的已安装环境。全量检查设置
`QT_QPA_PLATFORM=offscreen`、`SMARTSOM_REQUIRE_LEARNING=1`、
`SMARTSOM_REQUIRE_MARL=1`、`SMARTSOM_REQUIRE_STUDIO=1`，数值线程为 1。
本任务要求的学习与 Studio 后端实际参与检查；跳过的可选检查不计为通过。
最终完整日志：`/tmp/smartsom-small-fullgate-final.log`。

## 验证范围

- 自动矩阵全部 2304 个有向条目与 Factory 静态最短路一致；48 个节点，最大 20 ticks。
- 手填非对称值、缺失值、未知节点、负值、不可达下一阶段和错误节点身份的启动前校验。
- 全零矩阵的无缺陷单件手算：9 ticks，三次装货、三次卸货，检查仍耗时；语义回放一致。
- 正行程精确到达、在途不改派、同端口排队、来源预约保留、满位目的地等待及容量释放。
- MD 编辑距离手算；精确耦合与独立 LP 对照；历史先比较后更新；同池重排及独立数据。
- 同质/异质名义能力与容量加权质量守恒，完整计算后统一 ceil，另报取整损失。
- 资源 PPO/DQN、中央 RLlib PPO/SB3 在自动及零矩阵下的真实更新和加载。
- 连续与恢复运行的动作、权重、优化次数一致；DQN 额外检查 replay 和 target 时钟。
- Machine 取 PPO 指定 update、Buffer 取独立 DQN last、Dispatcher 从 ZIP 加载的混合评估。
- 批量跳过完成项、显式重试失败项、保留失败初始对照证据、训练 V 与测试 V 分别汇总。
- 新字段使用历史默认值时保留旧固定配方身份；实际启用矩阵、倍率或新取整语义时身份不同。
- 原有配置、学习工作流、回放及 Studio 的全量回归。

短训练验收采用 PPO 64 ticks、DQN 256 ticks，参数仅为触发真实更新而缩小；
没有用这些结果判断策略优劣。Qt 检查使用 offscreen，未进行人工桌面视觉验收。

## 交付的冻结准备

位置：`runs/prepared/small_hv`；24 组为 H0/H1 × Low/Mid/High × PPO/DQN × Auto/Zero。
每组 16384 训练 ticks、256 ticks/update，每四次更新验证五个独立案例，最后五个测试案例。
总计 393216 训练 ticks，另有 1920 个 validation episode、120 个模型测试 episode；
初始、规则和合法随机参照单独执行。当前所有子实验均待运行。

- H0 = 0.0；H1 = 0.18957188610128875。
- 训练池 V：Low = 0.24907836357752483；
  Mid = 0.2856591431299845；
  High = 0.37164465347925824。
- 11 个独立池（train 1 / validation 5 / test 5）；每池 64 件，同一池在条件间配对。
- 最终实现身份：`fbf87e9bd334f88024fd16e409dbecafeb0e0e8cf4ca217636abf250c8cf52b8`。

三档 V 是预先测量候选排列后选出的相对档位，不是通用阈值。
旧准备版本作为开发产物保留，不作为最终运行入口。
本记录证明工程行为；24 组尚未运行，不包含学习效果结论或正式研究验收。

文件职责、配置位置、模型组合和完整命令见 [Small 使用指南](small-hv-workflow.md)
及 [可组合策略工作流](composable-workflow.md)。


## Progress and process concurrency follow-up — 2026-09-26

The interrupted serial development run remains at
`runs/prepared/small_hv/experiments/20260926T141735Z-h0_low_dqn_auto-f0dbfdbcdd`,
with recovery `update-000016` at 4096 ticks. SIGINT was delivered to PID 14800;
the study lock was released and study/run manifests marked interrupted.
No original frozen input or checkpoint was rewritten.

Default runtime follows the archived PPO calibration at
`/Users/jianni/code/SmartSOM-week-2026-w38/artifacts/overnight-sl-screen/20260921T023344-b7a4af22/calibration.json`:
one environment, zero sampling subprocesses, one numerical thread, CPU, eight
independent experiments. Old aggregate adapter-decision rates were 164.28/s
(single), 530.01/s (four), 903.08/s (eight); process sampling was slower.
These are historical measurements, not v3 throughput predictions.

Fresh activated-environment verification:

- 63 progress/orchestration checks passed, including actual six-process refill,
  eight-process execution, mixed PPO/DQN updates, validation, evaluation, shared
  controls, completed-run skip, SIGINT worker cleanup and checkpoint continuation.
- 42 additional workflow/configuration/learning/display-invariance checks passed,
  one existing check skipped, two framework environment-registration warnings.
- Ruff check and format check passed across the worktree; git diff check passed.

The real process checks use small budgets and do not launch the 24-run study.
A newly prepared `runs/prepared/small_hv_parallel` freezes the changed source
and max_concurrent=8 while retaining the original 16384 ticks/256 update cadence,
5 validation cases every four updates, 5 held-out cases, and seed roots 101/303/202.
The user launches the new study. The old run cannot resume under a different
implementation identity; no source identity is silently relabeled.

## 2026-09-26 Rich study display repair

The interrupted `small_hv_parallel` remains untouched, including all eight recovery
checkpoints at 1,024 training ticks. Display changes are in telemetry/runtime.py
and the study coordinator's polling cycle: coalesced worker updates, fixed screen
slots, Rich alternate-screen overwrite, at most one ordinary redraw per second,
and no redraw for unchanged task state. Training and current validation/evaluation
case bars retain distinct physical counters. Unknown counters are static N/A.

Fresh checks: runtime-display/study-execution unit suite 66 passed; real parallel
study integration suite 3 passed (6/8 concurrent workers and interrupt/resume).
Repository Ruff lint/format and diff-whitespace checks passed. The terminal-output
regression asserts one frame for eight worker updates, cursor-home overwrite
without per-line erase/clear controls, and restoration of the original screen.
This is ANSI protocol verification; the user's terminal's rendered flicker still
needs confirmation after manual launch. No simulator or learning parameter changed.

A new `runs/prepared/small_hv_parallel_stable` bundle is used for manual restart;
old frozen identities are not rewritten to bypass implementation validation.
The full 24-case user run is not automatically launched by this repair.

## 2026-09-27 labelled study dashboard and overall ETA

The previous `small_hv_parallel_stable` study was stopped with SIGINT; its eight
1,024-tick recovery checkpoints remain intact. The display now has one labelled
row per active condition and a top workflow bar, elapsed time and approximate ETA.
Detailed group counts and PIDs remain in the presentation snapshot/logs.
`telemetry/study_progress.py` accounts training, scheduled validation, model test,
initial control and shared rule/random horizons; it reads result files without
loading models or altering simulator/learner state. Ended-case flags avoid counting
a finished case's last tick again as a new active case. Completed model-test paths
are published before controls so their progress is retained during later phases.

Fresh verification: 75 unit tests passed (runtime/study execution/overall progress);
3 real parallel tests passed (6/8 workers and interrupt/resume); repository Ruff
lint/format passed with 390 Python files; diff whitespace check passed. ANSI overwrite
regression still passes. A Rich SVG was rendered with Qt SVG and visually inspected
as PNG; this is a saved-counter layout preview, not live experiment evidence.

`small_hv_dashboard` was prepared but not started: 24 entries, max_concurrent=8,
current implementation identity verified, all 11 dataset JSON files exactly equal
to `small_hv_parallel_stable`. Budgets, root seeds, model settings and physics were
not changed. The user launches the full development study manually. No commit.
