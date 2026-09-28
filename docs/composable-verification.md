# 可组合策略 v3：工程验收记录

2026-09-26；实现分支 `codex/composable-policies-v3`。
基于 `517bf53dd2093a176896b371c25f0cd10cbc80cd` 的独立 worktree，改动未提交。
所有训练与评估均为临时目录内的开发验收；没有启动正式研究实验。

## 已完成的检查

| 检查 | 新完成结果 |
|---|---|
| `SMARTSOM_REQUIRE_LEARNING=1 SMARTSOM_REQUIRE_MARL=1 uv run --no-sync pytest -q` | 1,688 passed，19 skipped，退出码 0；这轮尚未安装 Qt |
| 安装锁定 Studio extra 后，`QT_QPA_PLATFORM=offscreen SMARTSOM_REQUIRE_STUDIO=1 uv run --no-sync pytest -q -m studio` | 128 passed，1,704 deselected，退出码 0 |
| `uv run --no-sync ruff check .` | 通过 |
| `uv run --no-sync ruff format --check .` | 通过 |
| `git diff --check` | 通过 |

计数是不同检查范围，不能直接相加。最终追加的 Machine 边界用例和 v3 Qt 回放用例，连同全部 v3 相关用例和扩展兼容检查，最终针对性回归为 **59 passed，退出码 0**。
运行环境为 Python 3.12.13、Torch 2.14.0、Ray 2.58.0、SB3/sb3-contrib 2.9.0、
PySide6 6.11.2；学习验收使用 CPU，Qt 使用 offscreen。

## 覆盖的行为

- 单一物理核、完整 tick 边界、双取货口共享现货和并行的一 tick 装货。
- START 后下一边界才贡献预约名额，出料不重复计数；现货不足与在途车辆不参与配货。
- 满位但兼容的目的地仍有候选，实际卸货检查容量；服务锁及未获服务的端口清让。
- 两套匹配规则、最优配对含车辆子集、均匀随机与独立可复现的平局流。
- 默认共享、稳定资源覆盖、不同 checkpoint 与 ZIP 伙伴、依赖副本和冻结统计。
- 资源 RLlib PPO/DQN、中央 RLlib PPO/MaskablePPO 的真实更新、加载与导出。
- PPO 的联合前缀概率、一次 clipping、物理时间 GAE；变长候选不截断及非法 Q mask。
- DQN 的 Δtick=0/1/5、终止与自举、replay 随机顺序、探索和 target 时钟。
- 连续运行与恢复运行的动作、权重、优化次数和 replay 状态一致。
- 无 Machine 决策时的零样本/零优化，以及手工 PRE 案例中的真实 Machine 更新。
- 评估录制、保存语义的执行审计，以及 Qt 的目标、预约、装货/卸货显示和准确 seek。
- 原有 v2 配置、物理协议和学习工作流的完整回归。

## 证据边界

旧权重没有被重新标记为 v3，使用新契约需要重训；组件组合仍要求工厂结构兼容。
默认规则可能交通阻塞或截断，完整组合的最佳 validation 不代表组件换伙伴后的最佳效果。
DQN 截断时，尚无合法下一决策的 owner 记录 censored 样本，不伪造 WAIT 或终止目标。

CUDA、真实桌面人工视觉验收、完整 H/V/Social Learning 和策略效果对比没有在这轮验证。
图形组合配置页按本轮范围留待后续。历史 checkpoint 全部保留；近期/周期设置维护索引，
不删除历史快照。操作入口与各文件的配置职责见 [使用指南](composable-workflow.md)。
