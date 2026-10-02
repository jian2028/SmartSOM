# 配置目录

`configs/` 外层用于你实际准备的新实验。当前保留的工厂是
[`factories/large.yaml`](factories/large.yaml)，来自 Studio 保存的 Large。
本次整理不创建新的任务、场景、算法或训练入口。

日常作者配置分为四类：`factories/`、`workloads/`、`algorithms/` 和
`runs/`（Experiment）。目前除已保存的 Large 外，日常区没有新实验输入。
Scenario、Composition 和 Policy 是运行前编译的内部对象，不需要单独
创建作者文件；批量组合由 Experiment 的 `matrix` 描述。

Studio 中 Template 7、8、9 的 AGV 数量保持 8、16、32；新增 Template 10、11、12
分别沿用这三张地图，AGV 数量为 10、20、40。现有实验配置不会自动切换模板。
内置模板仍位于
`src/smartsom/studio/templates/`，没有迁移或重新导出。

历史演示和工程验证配置统一放在 [`test/`](test/README.md)。它们是可复用的
验证案例，不是当前实验计划；不要把 `configs/test/runs/` 与仓库根目录自动
生成的 `runs/` 混淆。

配置分工：Factory 描述地图、资源和机器可靠性；Workload 描述固定任务或
可复用的任务生成规则；Algorithm 描述各 Agent 的规则或学习方法；
Experiment 引用前三者，并确定任务、运行设置和输出。
引用路径相对于写引用的配置文件。

## 产物与历史记录

- 根目录 `runs/`：后续新运行生成的结果目录；当前不预先创建实验。
- `backup/2026-09-26/`：此次归档的历史记录和处置清单，不纳入 Git。
- `artifacts/`：暂时保留的已注册源码工作树及符号链接，不是新的训练结果。

历史模型已列入删除范围。保留下来的日志、指标、回放、原始清单和配置快照
不代表模型仍可加载；历史记录的处置状态以归档中的 `retention.json` 为准。
