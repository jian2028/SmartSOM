# Implicit Social Learning (ISL)

当前唯一研究入口：Small/Medium × Base、ISL、Central、Rule。配置日期 2026-11-06。
Central 明确为 **E1a-current PPO**，不是 Morris，也不是历史 E1a 数值复现。
这里是实验定义，不是已运行的研究结果；本次迁移不启动实验。

## 矩阵与预算

| 规模 | 方法 | 训练种子 | 模型数 | 最终评估 episodes |
|---|---|---|---:|---:|
| Small：8 machines / 8 AGVs | Base / ISL / Central，各自一行入口 | 各 101 / 102 / 103 | 各 3 | 各 120 |
| Medium：20 machines / 20 AGVs | Base / ISL / Central，各自一行入口 | 各 101 / 102 / 103 | 各 3 | 各 120 |
| Small | Rule | 不训练 | 0 | 40 |
| Medium | Rule | 不训练 | 0 | 40 |

合计 **18 个模型、2 个 Rule 评估任务、800 次最终评估 episode**。
每个模型训练 1000 个完整 episode，按 `normal → low → medium → high`
均衡轮换，各档 250 次；四档共同训练一个模型，Small/Medium 分别训练。
最后每档评估 10 个固定实例，即每个模型 40 次；同规模所有方法和模型种子
使用相同实例。101/102/103 是模型与训练序列种子，不替换测试环境/动作种子。

## 实际文件结构

路径相对于开发工作树根目录；只列出定义本批实验的文件。

```text
experiments/implicit-sl/
├── README.md
├── small/
│   ├── base/experiment.yaml
│   ├── isl/experiment.yaml
│   ├── central/experiment.yaml
│   └── rule/experiment.yaml
└── medium/
    ├── base/experiment.yaml
    ├── isl/experiment.yaml
    ├── central/experiment.yaml
    └── rule/experiment.yaml

configs/
├── scenarios/machine-breakdown/
│   ├── small-320.yaml
│   └── medium-320.yaml
└── algorithms/
    ├── base-ddqn-rules.yaml
    ├── isl-ddqn-rules.yaml
    ├── e1a-current-ppo.yaml
    └── rule.yaml
```

不再有单独的 authored Case JSON。测试清单直接写在每个 Experiment 的 `test`
中；编译器将物理世界和校验值自动冻结在运行输入中，不需要人工维护运行产物。
同规模四个入口的内嵌面板应完全一致，测试会检查这一约束。
其他通用算法、场景和工程示例仍保留，不属于本批任务。

## 参数在哪里设置

先读[配置修改指南](../../docs/configuration-guide.md)。学习入口用 YAML 标量锚点
让最终评估节点跟随文件中的训练预算；CLI 覆盖不会动态重算锚点。


| 参数 | 文件及字段 |
|---|---|
| 工厂布局、机器、车辆、订单到达 | Scenario 的 `factory`、`jobs` |
| 故障和修复分布 | Scenario 的 `factory.machines.defaults.breakdown` |
| 场景、方法引用 | Experiment 的 `scenario`、`algorithm` |
| 训练种子 | Experiment 的 `train_seeds: [101, 102, 103]` |
| 每模型预算 | Experiment 的 `train_episodes: 1000`、`budget_boundary: complete_episode` |
| 训练四档、均衡轮换 | Experiment 的 `training_conditions`、`condition_sampling: balanced_cycle` |
| 测试档位、每档数量 | Experiment 的 `test.conditions`、`test.cases_per_condition: 10` |
| 固定环境与动作种子、世界校验值 | Experiment 的 `test.cases[]`：`id`、`condition`、`seed`、`action_seeds`、`scenario_sha256` |
| 最终 checkpoint 与 argmax | Experiment 的 `evaluation.checkpoints`、`run.selection` |
| 学习率、网络宽度、PPO/DDQN参数 | Algorithm 的学习策略组件参数 |
| 观测、网络、动作、奖励 | Algorithm 引用下表中的 Python 组件 |

测试总数由内嵌清单推导，不再另写 `test_cases: 40`。数量不匹配、重复ID/种子、
非法档位、世界校验不匹配均报错。内嵌测试面板不支持 controlled recovery，
该模式独立生成干预世界；不兼容组合会在编译时明确拒绝。completion 模式按
实际校准 drain 与 multiplier 构造保护时域，原世界 pin 与新时域不符也会拒绝。改变场景后不能沿用旧世界校验值假装同一测试集。

## 四档 machine breakdown

一个 Scenario 定义完整故障参数；运行时 condition 选择档位，不需要四份重复文件。

| condition | 故障间隔 scale / lower / upper（ticks） |
|---|---|
| normal | 不生成故障 |
| low | 600 / 60 / 1800 |
| medium | 300 / 30 / 900 |
| high | 150 / 15 / 450 |

维修共用 `scale=60, lower=10, upper=180`。采用条件截断指数分布，scale 不等于
截断后的实际均值。两个规模都是 320 订单、初始40、平均到达间隔14.4 ticks，
沿用现有路线/加工/交期分布；最后到达后2000 ticks为保护边界。

## Python 实现与比较边界

| 内容 | 实际实现 |
|---|---|
| Base 观测入口 | `src/smartsom/representations/base/observation.py` |
| Base 字段、统计和归一化 | `src/smartsom/observations/base.py` |
| ISL 入口与附加社会信息 | `src/smartsom/representations/social/observation.py`、`src/smartsom/observations/social.py` |
| Base 可学习编码、打包 | `src/smartsom/representations/base/{encoder,features,batching}.py` |
| ISL 编码入口与 attention | `src/smartsom/representations/social/encoder.py`、`src/smartsom/learning/social_actor.py` |
| Base/ISL 网络与 DDQN | `src/smartsom/learning/{models,heads,ownership}.py`、`src/smartsom/learning/dqn/{learner,heads,replay}.py` |
| Base/ISL 规则候选动作映射 | `src/smartsom/algorithms/rule_actions.py` |
| Base/ISL/Rule 默认奖励 | `src/smartsom/rewards/default.py` |
| Central actor/critic、网络、reward | `src/smartsom/controllers/e1a/{observation,network,reward}.py` |
| Central 合法动作与执行 | `src/smartsom/engine/central_control.py` |
| PPO/GAE | `src/smartsom/learning/ppo/learner.py`、`src/smartsom/learning/returns.py` |
| Rule 观测与规则 | `src/smartsom/observations/rules.py`、`src/smartsom/policies/rule.py` |
| 共用 mover | `src/smartsom/traffic/rule_mover.py` |

Base/ISL 保持同角色共享、native 初始化、width32、DDQN、规则候选、禁用 STOP/
EMPTY_WAIT，及现有学习率/奖励等配置；本次只改组织，不改两者的输入和动作。
Base/ISL 默认奖励为 `-(逾期积分 + 1000 × 不合格交付数) / 320`，learner 再乘 .001。
Central 沿用 E1a-current PPO：64×64 tanh、GAE、独立 E1a reward，reference_jobs=128、
reference_ticks=4096 仍为明确的归一化参考参数，并非本次订单数量/episode预算。
Rule 沿用默认 Rule 配方和动作权限。Central/Rule 不等于只切换 Base 的决策机制；
它们的动作空间、学习算法或奖励定义不同。跨方法比较物理交付/迟交/合格率，
不要直接把不同定义的学习 return 当成同一个指标。

学习方法每50 episode保存，只在1000 episode最终argmax评估，无中间验证；
`train_ticks` 保留但完整episode预算优先。Rule 直接evaluate，无模型训练。
全部使用 configured 方法，无额外自动追加基线。Medium stage0，Small stage1，
最大并发2；保留资源保护、W&B online/SmartSOM、无科学wall-clock截断和无自动重试。

## 检查与来源

只读检查全部20个任务，不启动训练或评估：

```sh
uv run smartsom check experiments/implicit-sl --json
```

可单独检查 `experiments/implicit-sl/small/base` 等目录。新结果会使用新科学配置身份；
已有 runs/checkpoint 不移动、不重写，历史任务使用冻结源码恢复。
正式实验仍需 integrated main 或明确 tag，并另行授权启动。

以下保留原独立 provenance 文件的内容作为来源说明，不是运行manifest：

```json
{
  "status": "STATIC_RECIPE_NOT_A_RUN",
  "inherited_protocol": "rule-dqn-confirm-20261101",
  "historical_sources": {
    "rule-dqn-confirm-20261101/source/configs/base-diagnostic/target-workload.yaml": "5297ecc63463b98a4163c68666d632cf28dc832fe76614021e7d270bd1f5947d",
    "rule-dqn-confirm-20261101/source/configs/rule-dqn/confirmation-plan.json": "baf7e132eafa8c69da44d8b0a4763d1e057e788915b9116ef65e28d7b4a31ced",
    "rule-dqn-confirm-20261101/source/configs/rule-dqn/scientific-contract.json": "3e59b83bbab34b2181b4c1e793ce009be35df71964d6b427464d20a237cb588c",
    "rule-dqn-server-20261101/manifest.json": "cc2500534042d4304b1eae12878086afb41b32053631426e4c18f653341485a5",
    "rule-dqn-confirm-20261101/source/configs/rule-dqn/training-worlds.json": "077bb702d4d31b60027a892ae723483aa48775e75e76581dffa94aec8bd5c12d"
  },
  "panel_identity": "Exact historical 40 case IDs, conditions, scenario seeds and action seeds; physical world hashes regenerated for each new map.",
  "maps": {
    "small-factory": "0e79e92aec8259eb21e6d42e5abc84c9f93133b0f932b60f54bbe1a1ad87c5ad",
    "medium-factory": "77a7281a24cde4ec951a43709a7f9b8625ffbc2d6fcecec80917885fc49bad51"
  }
}
```
