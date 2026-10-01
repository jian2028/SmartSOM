# 演示与工程验证配置

这里保存历史演示和工程验证输入。2026-09-26 曾从日常区迁入 195 个配置；
之后又增加了 v3 工程案例。旧作者格式正在按
[`处置清单`](../../docs/validation/legacy-author-disposition.csv)逐项迁移或退役，
因此不能把整个目录当作新四文件接口的示例。

| 目录 | 配置数 | 用途 |
| --- | ---: | --- |
| `factories/` | 24 | 手算案例、模板演示及参考工厂 |
| `workloads/` | 44 | 固定任务和任务生成案例，含新 v3 案例 |
| `algorithms/` | 30 | 规则及学习参数案例，含新 v2 案例 |
| `runs/` | 66 | 演示或验证入口，含新 v4 案例 |
| `scenarios/` | 46 | 待迁移的旧物理配方 |
| `studies/` | 6 | 待审计的旧批量案例 |
| `compositions/`、`policies/`、`rules/`、`transport/` | 30 | 旧接口依赖资料 |

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
