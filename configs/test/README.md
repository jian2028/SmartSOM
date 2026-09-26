# 演示与工程验证配置

这里保存从原 `configs/` 分类目录迁入的 195 个既有配置：

| 目录 | 配置数 | 用途 |
| --- | ---: | --- |
| `factories/` | 22 | 手算案例、模板演示及参考工厂 |
| `workloads/` | 41 | 固定任务和任务生成案例 |
| `scenarios/` | 44 | 到达、加工时间、故障、缓冲区和质量验证 |
| `algorithms/` | 27 | 规则、脚本控制器及学习参数案例 |
| `runs/` | 56 | 演示或验证的启动入口 |
| `studies/` | 5 | 批量验证案例 |

案例内部的工厂、任务和算法引用仍按分类组织；指向仓库数据或输出目录的
相对路径已适配新位置。迁移没有把旧格式改写成新格式，也没有重算历史结果。

从仓库根目录检查已有演示（检查命令不启动仿真）：

```sh
uv run --no-sync smartsom validate --config configs/test/runs/template1_static.yaml
uv run --no-sync smartsom show-config --config configs/test/runs/template1_static.yaml
```

CP-SAT 配置作为历史案例保留；当前网格运行仍没有 CP-SAT adapter。
需要新实验时在外层 `configs/` 明确选择并编写配置，不自动沿用旧模型。

历史文档路径对照见
[`docs/validation/config-path-migration.md`](../../docs/validation/config-path-migration.md)。
