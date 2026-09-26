# 可组合策略：文件职责与使用入口

新版使用 `smartsom.experiment-config/v3`。Machine、Buffer、Dispatcher、Mover
可以分别选择不同实验、不同 checkpoint、不同网络的兼容组件，也可以分别使用规则。
同类资源默认共享一个策略组，允许按资源 ID 覆盖。中央模式只有一个完整控制器，
不能拆成独立角色模型。图形组合配置页后续建设，本轮使用 YAML、CLI 和 Python API。

## 每个文件控制什么

所有手写配置分类放在 `configs/`；无需为每个实验创建项目目录。

| 目录 | 在这里设置 |
|---|---|
| `factories/` | 工厂、资源 ID、端口、容量、加工模式 |
| `workloads/` | Job 路线、加工时间、订单或生成分布 |
| `scenarios/` | 引用工厂和工作负载；到达、扰动、物理时限 |
| `policies/` | 一个角色的规则、新模型网络或已有模型来源 |
| `compositions/` | 命名共享权重组、四角色绑定、资源覆盖、配货规则 |
| `rules/` | `global_optimal` 或 `priority_greedy` 配货 |
| `algorithms/` | PPO / DQN 优化参数，不放角色绑定和模型路径 |
| `runs/` | **启动入口**：场景、组合、训练组、物理预算、三套种子 |

仓库根目录的 `runs/` 是自动生成的产物目录，与 `configs/test/runs/` 分开。
引用相对于**写引用的文件**；例如组合中的 `../policies/machine_rule.yaml`
相对于 `configs/test/compositions/`。执行前解析 `best/last` 为确定 update，
归档冻结伙伴模型，训练时不再读取手写配置。

生成带名字前缀的一套分类文件，已有文件冲突时会报错：

```sh
smartsom init composable configs --name my_trial
smartsom show-config --config configs/test/runs/my_trial_train_machine_ppo.yaml
```

它生成 `policies/my_trial_machine_new_ppo.yaml`、
`compositions/my_trial_train_machine_ppo.yaml` 等分类文件，不自动开始训练。

## 从哪里启动

使用已安装的环境，先激活：

```sh
source .venv/bin/activate
```

Small H/V 与运输矩阵的完整配置和批量入口见 [Small 工作流](small-hv-workflow.md)。

```sh
# 先检查实际绑定、模型身份、训练/冻结状态、兼容性、预算和种子
smartsom show-config --config configs/test/runs/train_machine_ppo.yaml

# 只训练 Machine，其余三个伙伴使用规则
smartsom train --config configs/test/runs/train_machine_ppo.yaml

# 同一任务统一选择算法，可以训练多个策略组
smartsom train --config configs/test/runs/train_all_ppo.yaml
smartsom train --config configs/test/runs/train_all_dqn.yaml

# 训练 + 途中 validation + 最后独立 evaluate
smartsom train-evaluate --config configs/test/runs/train_machine_ppo.yaml

# 中央完整控制器
smartsom train --config configs/test/runs/central_rllib_ppo.yaml
smartsom train --config configs/test/runs/central_sb3_ppo.yaml

# 默认沿用原训练组合与独立测试输入
smartsom evaluate RUN_DIRECTORY --checkpoint best
smartsom evaluate RUN_DIRECTORY --checkpoint update-000004

# 纯规则或自己的混合组合
smartsom evaluate --config configs/test/runs/evaluate_all_rules.yaml

# 完整恢复原组合和继续状态
smartsom resume RUN_DIRECTORY
```

`total_ticks` 累计所有逻辑环境实际推进的物理 tick。一个 tick 有多个 Agent、
Buffer 选择多件货都不额外消耗预算。最后一个更新可以使用剩余完整 tick；
更新、验证和保存只在物理边界发生。另报各组决策、样本和实际优化次数。

`runtime.num_envs` 创建独立逻辑环境；`sampling_processes` 使用有序物理采样进程，
按固定环境轮转归并并同步策略和随机状态，优先保证可恢复顺序，不承诺增加进程一定加速。
`numerical_threads` 限制数值线程，`device` 选择实际模型设备。

## 如何组合不同模型

`configs/test/policies/machine_from_A.yaml`：

```yaml
schema: smartsom.policy/v1
role: machine
implementation:
  kind: model
  model:
    source: ../../runs/EXPERIMENT_A
    checkpoint: best
    group: machine
```

Buffer 可以选 B 的 `update-000008`，Dispatcher 选 C 的 `last`，Mover 选 ZIP。
不同角色不必来自同一实验、同一算法或同一网络；必须匹配角色、工厂结构与
v3 动作/观测契约。初始化可训练组时算法与后端还须匹配。
模型包携带网络与编码设置，引用文件不能重新定义它们。

固定 ZIP 已是确定组件，不再选择 checkpoint：

```yaml
schema: smartsom.policy/v1
role: mover
implementation:
  kind: model
  model:
    source: ../../models/mover_D.zip
```

组合文件 `configs/test/compositions/compare_ABCD.yaml`：

```yaml
schema: smartsom.composition/v1
pickup_matching: ../rules/pickup_global_optimal.yaml
groups:
  machine_A: {policy: ../policies/machine_from_A.yaml}
  buffer_B: {policy: ../policies/buffer_from_B.yaml}
  dispatcher_C: {policy: ../policies/dispatcher_from_C.yaml}
  mover_D: {policy: ../policies/mover_from_D.yaml}
bindings:
  machine: {default: machine_A}
  buffer: {default: buffer_B}
  dispatcher: {default: dispatcher_C}
  mover: {default: mover_D}
```

实验入口 `configs/test/runs/comparison.yaml`：

```yaml
schema: smartsom.experiment-config/v3
scenario: ../scenarios/template1_static.yaml
composition: ../compositions/compare_ABCD.yaml
seed: 101
evaluation:
  seed: 202
  replications: 5
output:
  root: ../../runs
  name: comparison_ABCD
```

独立评估没有 `training` 段。没有已有权重的 `new_model` 不能当冻结伙伴。

## 如何只训练指定资源或组

策略组是权重共享单位。例如所有机器默认使用 `machine_shared`，
机器 `machine_001` 另用 `machine_special`：

```yaml
groups:
  machine_shared: {policy: ../policies/machine_from_A.yaml}
  machine_special: {policy: ../policies/machine_new_ppo.yaml}
  # 其余角色也需要在 groups 与 bindings 中声明
bindings:
  machine:
    default: machine_shared
    overrides: {machine_001: machine_special}
```

实验中 `training.groups: [machine_special]` 只更新覆盖组。
默认组须为已有模型或规则。即使两组从同一个模型初始化，也拥有独立优化器、
探索流和训练状态。冻结伙伴的权重与归一化统计均保持不变。

默认编码保留公开的下一操作类型、剩余路线、各兼容机器的 nominal time、资源/端口身份、库存、容量和预约。候选维度由冻结工厂契约推导；批内 padding 不截断候选或前缀。

PPO 的 actor/critic 和 DQN 的 Q 设置只写在新模型策略文件：
`implementation.extensions.network.actor/critic` 或 `.q`。
观测扩展写在 `implementation.extensions.observation`。
奖励扩展保留在 `training.reward`；team 每物理 tick 变换一次，再按角色变换。
角色名称为 `machine_policy`、`buffer_policy`、`dispatcher_policy`、
`mover_policy`。规则与模型遵守同一候选和物理限制。

有状态观测扩展应遵守 `PublicObservation.update_statistics` 或
`set_training(False)`。冻结实例若更新统计，会恢复统计并立即报错。

## Validation、evaluate 与 resume

- Validation 使用当前权重、相同伙伴和配货规则，以及 seed 303 的冻结输入。
- `best` 是完整组合的最佳验证快照。导出 Machine 会保留伙伴来源，不代表换伙伴后仍最佳。
- 默认最后 evaluate 使用 seed 202 的独立输入；可以先只 train，再手动 evaluate。
- 换伙伴或配货规则后评估，写新的 composition/run；结果记录实际组合。
- 换伙伴后继续训练是新实验初始化，在策略文件里选择已有模型。
- `resume` 恢复原组合、输入配方、模型、优化器、轨迹和随机状态，
  DQN 还恢复 replay、online/target、探索和 target 时钟。

PPO 的 Buffer 前缀概率求和，每 owner/物理边界只形成一次 ratio/clipping。
DQN 前缀中间选择 `Δtick=0`，奖励使用实际区间折扣。截断时对当前边界实际
存在的合法下一决策自举；如果 owner 尚无合法决策（如加工未结束），未闭合
样本记录为 censored，不伪造 WAIT 或终止目标。没有决策的组如实记录零样本。

## 产物、导出和回放

```text
RUN_DIRECTORY/
  run.json                      源码、组合身份、预算、样本/更新、状态
  config/prepared.json           完整冻结配方与 validation/evaluate 输入
  dependencies/                 冻结伙伴副本
  checkpoints/
    last.json / best.json        指向具体 update
    recovery.json               最近完整恢复边界
    recent.json                 最近 keep_last 个快照的索引
    periodic-*.json              every_updates 的周期索引
    update-000004/
      snapshot.json
      continuation.pkl          完整本地实验继续状态
      groups/GROUP/
        model.json               网络、编码、契约、算法、工厂、伙伴来源
        weights.pt
        encoder.json
      controllers/central/      中央组合整套控制器
  logs/validation-*.json
  evidence/actions.json
  reports/training.json
  evaluation/.../summary.json
```

历史快照全部保留，`keep_last` 控制近期索引，`every_updates` 控制周期索引。
`best` 要启用并实际完成 validation。便携组件是权重与 JSON；
完整继续状态是本地 Python 实验状态。

```sh
smartsom export RUN_DIRECTORY --group machine --checkpoint best --output models/machine_A.zip
smartsom export CENTRAL_RUN --checkpoint last --output models/controller.zip
smartsom export RUN_DIRECTORY --kind experiment --output exports/full_experiment.zip

smartsom playback EVALUATION_DIRECTORY/evidence/case-0000
smartsom audit EVALUATION_DIRECTORY
```

评估每个 case 是标准录制。回放保留目标、预约、装卸阶段与语义裁决；
装卸开始在 `boundary_state`，完成在本 tick 的 `state`。质量由环境自动检查。

评估分别报告 reward、交付、flow time、来源预约/目的地等待、makespan、截断和异常。
这里 waiting 是运输等待计数，不能当作全部生产队列总等待。
全规则最短路遵守端口清让与冲突约束，没有死锁救援；截断会如实报告。
工程通过不能证明学习优于规则。

## Python API

```python
from smartsom import api

config = api.load_config("configs/test/runs/train_machine_ppo.yaml")
print(api.show_config(config))
result = api.train_evaluate(config)
api.evaluate(result.training.run_dir)

comparison = api.load_config("configs/test/runs/comparison.yaml")
api.evaluate(config=comparison)
api.export_model(
    result.training.run_dir, "models/machine.zip", group="machine", checkpoint="last"
)
api.resume(result.training.run_dir)
```

`api.prepare(config)` 返回冻结的 `PreparedComposition`，
`api.train_prepared(prepared)` 不再读取手写配置。

## 显式迁移

```sh
smartsom migrate configs/test/runs/OLD.yaml --to v3 --output configs/test/runs/NEW.yaml --preview --total-ticks 4096 --ticks-per-update 256
```

先看差异；去掉 `--preview` 才写新文件。旧文件和旧产物保留。
旧 steps 不自动换算物理 ticks。资源迁移默认 Machine 新模型加规则伙伴，
训练更多组时显式编辑组合。旧 AGV 权重不拆成 Dispatcher/Mover，要求重训。

见 [架构](architecture.md) 与 [ADR 0020](decisions/0020-composable-policies-and-physical-tick-learning.md)。


## 代码文件的职责与验收记录

| 文件 | 修改这处会影响什么 |
|---|---|
| `domain/production_decisions.py` | 四角色的语义候选、提议和 v3 契约身份 |
| `engine/production.py` | 唯一物理核心；库存、加工、自动检查、冲突及时间 |
| `engine/production_protocol.py` | 分阶段边界、来源预约、装卸服务、容量与端口清让 |
| `algorithms/production_composition.py` | 策略组路由、资源覆盖、条件前缀、保存提议及回放 |
| `algorithms/pickup_matching.py` | 两套配货规则、精确 DP、独立平局随机流 |
| `config/policies.py`、`compositions.py`、`experiment_v3.py` | 三层 YAML 校验、相对路径、兼容性和启动前冻结 |
| `learning/production_models.py`、`production_inference.py` | 公开观测编码、网络扩展、候选打分和组件推理 |
| `learning/production_collection.py` | owner 轨迹、物理时间 GAE、前缀样本与 replay |
| `learning/production_rllib_v3.py`、`production_sb3_v3.py` | RLlib PPO/DQN 与中央 MaskablePPO 的实际优化 |
| `experiments/composable.py` | 训练预算、验证、评估、产物、导出和完整恢复 |
| `api.py`、`experiments/cli.py` | 用户执行入口 |
| `studio/replay_inspector.py`、`trace/production.py` | 新边界字段展示和保存语义的执行审计 |

路径均相对于 `src/smartsom/`。常规使用只编辑配置；新增策略或学习接口时才修改代码。

见 [工程验收记录](composable-verification.md)。这些检查证明接口、更新与恢复行为，
不证明学习效果优于规则。正式研究实验仍须遵守仓库的源码冻结和实验记录要求。
