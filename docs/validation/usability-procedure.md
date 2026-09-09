# macOS 使用体验验收流程

本流程生成新的工程证据，历史第 12/13 项的配方、报告和失败记录保持原样。
实际通过状态和实现提交另见完成记录；本流程文件本身不表示验收通过。

## 运行来源与依赖

实现先集成本地 `main`，再从该提交建立保留的 detached 工作区。
工作区和输出放在持久的 `artifacts/usability/formal-<commit>/` 下。
使用该工作区自己的 Python 3.12 环境，避免 editable 安装导入另一个 checkout：

```sh
git worktree add --detach artifacts/usability/formal-<commit>/source <commit>
cd artifacts/usability/formal-<commit>/source
uv sync --locked --extra learning --extra cpu --extra cp --extra reports \
  --extra tensorboard --extra wandb --extra search
uv run --no-sync python -c 'import smartsom; print(smartsom.__file__)'
uv run --no-sync python scripts/validate_usability.py --output-dir ../evidence
```

脚本拒绝覆盖已有目录，默认要求 macOS、Python 3.12、干净源码且提交已集成本地
`main`。开始和结束均核对提交，实际包必须来自当前工作区。报告记录锁文件摘要、
已安装依赖、平台、各阶段日志和真实产物目录。开发调试可明确使用 `--development`，
其报告不能作为正式通过记录。

## 验收内容

- 从历史配置重新训练集中式 RLlib 4096 步、SB3 1024 步，以及资源 MARL
  seed101／4096 联合轮／49152 agent steps／16 次更新。源码、完整输入和算法配方
  均受固定摘要检查；训练结束后再次检查实际快照。
- 第 12 项执行 15 次评估，每个 replication 恰有 SPT、RLlib 和 SB3 各一次，
  使用冻结的 seed202、完整场景输入和模型参数，完成全部轨迹与观察回放。
- 第 13 项检查双角色参数更新、保存恢复、训练审计和价值损失，再执行 seed202
  的五对 MARL/SPT；十次完工和物理回放、五次联合回放全部通过才接受。
- 三后端真实测试覆盖完整续训、单环境和多进程采样、状态化扩展、奖励隔离、
  后端探测、公共 API、模型 ZIP 直接评估、完整实验搬迁恢复，以及批量和搜索。
  TensorBoard、W&B 离线、报告静态导出和离线事件控件也在固定测试清单中。
- 执行测试前独立收集完整 nodeid 清单，清除外部 pytest 筛选项；XML 必须与
  清单逐项匹配，没有漏项、跳过、失败或错误。证据通过显式 `--basetemp` 保留。

全仓回归、Ruff、格式和锁检查在正式来源提交前完成。CI 配置随能力更新，
本地结果不代表远端 CI 已执行。W&B 不向云端发布。

工程缺陷采用新提交向前修复，并从新的固定来源重新验收。学习停滞或数值异常先
保留证据和诊断，不自动换 seed、改变预算或挑选 checkpoint。
特意构造的失败测试通过，仅表示状态和证据处理符合约定，不表示对应策略完工。

Linux、h20、CUDA 安装和训练仍待单独实测。HTML 的离线 DOM 控件测试与
PNG/SVG/PDF 导出不等于真实浏览器的视觉和下载验收；当前浏览器自动化对本地
`file://` 的拒绝记录保留，未使用其他入口绕过。
