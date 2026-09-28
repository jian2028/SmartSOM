# 配置目录

`configs/` 外层用于你实际准备的新实验。当前保留的工厂是
[`factories/large.yaml`](factories/large.yaml)，来自 Studio 保存的 Large。
本次整理不创建新的任务、场景、算法或训练入口。

日常配置区保留完整分类：`factories/`、`workloads/`、`scenarios/`、
`algorithms/`、`runs/` 和 `studies/`。目前除已保存的 Large 外，其余分类
留空，仅放置 `.gitkeep`，以便 Git 保留目录；尚未填写新的实验配置。

Studio 中的最新 Small、Medium、Large 是 Template 7、8、9；内置模板仍位于
`src/smartsom/studio/templates/`，没有迁移或重新导出。

历史演示和工程验证配置统一放在 [`test/`](test/README.md)。它们是可复用的
验证案例，不是当前实验计划；不要把 `configs/test/runs/` 与仓库根目录自动
生成的 `runs/` 混淆。

配置分工：Factory 描述工厂；Workload 描述任务；Scenario 引用工厂和任务，
配置到达、扰动与物理时限；Algorithm 选择已有控制器；Run 是启动入口。
引用路径相对于写引用的配置文件。

## 产物与历史记录

- 根目录 `runs/`：后续新运行生成的结果目录；当前不预先创建实验。
- `backup/2026-09-26/`：此次归档的历史记录和处置清单，不纳入 Git。
- `artifacts/`：暂时保留的已注册源码工作树及符号链接，不是新的训练结果。

历史模型已列入删除范围。保留下来的日志、指标、回放、原始清单和配置快照
不代表模型仍可加载；历史记录的处置状态以归档中的 `retention.json` 为准。
