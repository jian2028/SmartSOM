# 使用体验重构：macOS CPU 验收记录

2026-09-10（Asia/Shanghai），实现
`4b4a7c2719d8c7632df5e5e1e90c7eb7d055605f` 已集成本地 `main`，
并通过新的固定源码自动化验收。总流程退出码为 0，报告状态为 `passed`。
本记录单独提交，不改变被验收的实现，也不替代历史第 12/13 项证据。

这是 macOS CPU 工程验证。Linux/h20/CUDA 仍待实测；HTML 的真实浏览器视觉和
下载验收仍未验证。学习效果没有超过 SPT，不据此提出算法优势或泛化结论。

## 固定来源与证据位置

- 验证工作区：[保留的 detached source](../../artifacts/usability/formal-4b4a7c2/source)。
  开始、结束均为同一干净提交，且该提交已集成本地 `main`。
- 实际导入包来自该工作区的 `src/smartsom/__init__.py`，解释器来自该工作区
  自己的 `.venv`，没有使用另一个 checkout 的 editable 安装。
- 平台：`macOS-26.6.2-arm64-arm-64bit`；Python `3.12.13`；CPU 数值线程 1。
- `uv sync --locked --offline` 安装 learning、cpu、cp、reports、tensorboard、
  wandb、search；[安装与导入记录](../../artifacts/usability/formal-4b4a7c2/setup.json)。
  Torch `2.14.0`、Ray `2.58.0`、Gymnasium `1.2.2`、PettingZoo `1.27.0`、
  SB3/sb3-contrib `2.9.0`；全部实际依赖版本见总报告。
- 锁文件 SHA-256：`72143eb522e3331aa1e4dce96630f597094901a04ffdb0cceecdf16c7aa3434d`。
- 正式流程时间：`2026-09-09T19:35:16+00:00` 至 `19:57:36+00:00`。
- [总报告 JSON](../../artifacts/usability/formal-4b4a7c2/evidence/report.json)，
  SHA-256：`38cb2ed06087b86cfa13f212b0599914dec2aba777aae7b1f8858fdc7cc4900a`。
  [执行日志](../../artifacts/usability/formal-4b4a7c2/acceptance.log)及各阶段日志均保留。

上述链接指向本机保留的证据，数据和工作区没有提交到 Git。源目录的实际位置为
`/Users/jianni/code/SmartSOM/artifacts/usability/formal-4b4a7c2/source`。
复验命令及严格拒绝条件见[验收流程](usability-procedure.md)。

## 冻结配方复验

三条路线均从相同固定提交重新训练，没有沿用开发 checkpoint，也没有把新
quickstart 的周期验证和模型选择默认值套入历史配方。

| 路线 | 实际采样 | PPO 更新 | 结果 |
| --- | --- | --- | --- |
| Centralized RLlib PPO | 4096 环境步 | 16 | 参数更新、保存加载、训练审计通过 |
| SB3 MaskablePPO | 1024 环境步 | 4 | 参数更新、保存加载、训练审计通过；后端原生 learner_updates 为 40 |
| Resource MARL PPO | 4096 联合轮／49152 agent steps | 16 | machine/AGV 两角色更新、保存加载、训练审计通过；learner reward scale 为 0.0001 |

[第 12 项报告](../../artifacts/usability/formal-4b4a7c2/evidence/item12/report.json)
包含 seed202 的 5 组 SPT/RLlib/SB3，共 15 次评估。
[第 13 项报告](../../artifacts/usability/formal-4b4a7c2/evidence/item13/report.json)
包含 seed202 的 5 对 MARL/SPT，共 10 次评估。
每个 replication 的实际完整环境输入一致，各方法恰出现一次。
25 次全部完工，物理动作、调度和观察回放通过；5 次 MARL 联合回放全部通过。
所有训练、评估和审计记录均指向同一干净实现。

| 策略 | replication 0–4 的 makespan | 均值 | 样本 SD |
| --- | --- | ---: | ---: |
| SPT | 120、135、147、135、136 | 134.6 | 9.6073 |
| Centralized RLlib PPO | 463、478、464、420、421 | 449.2 | 26.8645 |
| SB3 MaskablePPO | 597、597、595、595、595 | 595.8 | 1.0954 |
| Resource MARL PPO | 233、225、205、205、205 | 214.6 | 13.4462 |

每种学习策略只有一个训练 seed，表中的五次评估不是五个独立训练样本。
MARL 训练账本有 163 个 episode：143 完工、19 `policy_stalled`、1 `training_budget_stop`。
训练预算完成与独立评估全部完工，不意味着训练期间每局都完工。

## 新功能与回归

固定源码首先收集完整 [164 项 nodeid 清单](../../artifacts/usability/formal-4b4a7c2/evidence/feature-collection.json)，
再与 [JUnit XML](../../artifacts/usability/formal-4b4a7c2/evidence/features.xml)逐项匹配：
**164 通过，0 跳过，0 失败，0 错误，927.92 秒**。
[测试日志](../../artifacts/usability/formal-4b4a7c2/evidence/logs/features.log)
记录 58 条框架弃用等警告；没有把警告或测试进程启动当作通过。
测试产物通过显式 `--basetemp` 保留在 `evidence/feature-evidence/`。

覆盖三后端完整恢复与 optimizer/RNG/活动环境对照、多环境与 spawn 采样、
状态化观测/网络/奖励扩展及回放、奖励隔离、实际后端探测、公共 API、ZIP 直接
评估、验证源文件删除和完整实验搬迁后的恢复、批量与网格/Optuna 搜索、
TensorBoard/W&B 离线、报告和导出。失败测试通过表示失败被正确识别和保留，
不表示被测失败策略完工。W&B 本次没有向云端发布实验。

| 检查 | 来源与结果 | 保留证据 |
| --- | --- | --- |
| 全仓完整依赖回归，包括 centralized/MARL/CP | `5c3a9bd`：2085 通过，0 跳过，1486.14 秒 | [日志](../../artifacts/usability/development/full-regression-5c3a9bd.log)、[XML](../../artifacts/usability/development/full-regression-5c3a9bd.xml) |
| 最后两项诊断修复后的聚焦验证 | `4b4a7c2`：31 通过，0 跳过，24.57 秒；覆盖显示、doctor、真实后端探测 | [日志](../../artifacts/usability/development/final-diagnostics.log)、[XML](../../artifacts/usability/development/final-diagnostics.xml) |
| 最终基础依赖全仓回归 | `4b4a7c2`：1639 通过，156 可选依赖测试按声明跳过，131.05 秒 | [日志](../../artifacts/usability/development/base-regression-4b4a7c2.log)、[XML](../../artifacts/usability/development/base-regression-4b4a7c2.xml) |
| 独立 wheel 基础安装 | 与 `5c3a9bd` 相同源码树；在仓库外运行 6 个 CLI 流程，JSP makespan 6，6 项审计通过；未安装或加载学习依赖 | [报告](../../artifacts/usability/development/wheel-base-5c3a9bd/verification/report.json)、[来源](../../artifacts/usability/development/wheel-base-5c3a9bd/verification/setup-and-source.json) |
| Ruff／格式／锁／差异空白 | 最终实现：Ruff 通过、270 文件格式通过、锁检查通过、`git diff --check` 通过 | 本地完成检查；未宣称远端 CI 已运行 |

全仓 2085 项运行后仅有诊断元数据与终端指标呈现两项修复；其聚焦测试及本次
固定源码 164 项清单重新通过。上述测试集合存在重叠，不合并成独立样本数。
随机搜索的配置展开、冻结与 RNG 隔离由开发回归覆盖，不计入正式 164 项清单。
CI 已加入相应能力检查，但本次没有 push 或运行远端 CI。

## 可直接查看和使用的产物

- [MARL 训练 HTML](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/marl-training.html)、
  [第 13 项评估 HTML](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/item13-evaluation.html)。
  历史 study 格式的 HTML 提供轨迹视图，配对数值以对应 JSON 报告为准。
- [甘特图 PNG](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/item13-model-timeline.png)、
  [SVG](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/item13-model-timeline.svg)、
  [PDF](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/item13-model-timeline.pdf)。
  [完整图表清单](../../artifacts/usability/formal-4b4a7c2/evidence/derived-reports/index.json)
  共 11 个产物，均来自实际训练或已审计事件；PNG 已实际查看。
- 最终模型：[MARL ZIP](../../artifacts/usability/formal-4b4a7c2/evidence/exported-models/marl-last.zip)、
  [RLlib ZIP](../../artifacts/usability/formal-4b4a7c2/evidence/exported-models/rllib-last.zip)、
  [SB3 ZIP](../../artifacts/usability/formal-4b4a7c2/evidence/exported-models/sb3-last.zip)。
  三包的真实成员及校验清单验证通过；[摘要索引](../../artifacts/usability/formal-4b4a7c2/evidence/exported-models/index.json)。
  模型包供评估或权重初始化，完整续训使用原运行或完整实验包。

HTML 离线 DOM 测试验证播放/暂停、同 tick 逐事件、过滤和缩放；静态导出也已
实测。当前浏览器工具拒绝打开本地 `file://`，没有改用其他入口绕过，因此
真实浏览器的视觉、交互和下载检查保留为未验证。

## 保留的失败及后续边界

128 步扩展示例是开发演示，不是冻结验收配方。SB3 和 centralized RLlib 的
单次评估完工；资源示例在首个联合决策全部 NOOP，评估状态为 `not_completed`，
原因为 `policy_stalled`，0/1 完工，物理动作数为 0。其扩展/奖励前缀回放为 `partial_verified`，
不能称为完整回放通过。没有更换 seed、追加预算或挑选其他 checkpoint。

[示例历史保存清单](../../artifacts/usability/development/extension-example-history/preservation.json)
记录逐文件原始位置和摘要；[三个实验包清单](../../artifacts/usability/development/extension-example-bundles/index.json)
保留实际结果和状态。缺依赖、离线缓存缺失及沙箱限制的先前尝试也保留，
没有删除失败后只展示成功。历史第 12/13 项报告及配方保持原样。

Linux/h20 安装、驱动、CUDA 训练及跨构建加载仍需独立实测。完整续训要求相同
源码身份与锁定依赖；后续文档提交不替代 `4b4a7c2`，需要恢复本次固定实验时
使用保留的固定源码工作区。治理文件、论文规划、外部周计划均未在本轮修改；
无 push、tag、历史改写或远端分歧处理。
