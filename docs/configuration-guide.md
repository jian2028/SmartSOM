# 怎样修改实验配置

日常从对应的 `experiment.yaml` 开始。它引用 Scenario（环境）和 Algorithm
（算法组件）；无需增加一份 Case 文件。具体入口见
[ISL 实验说明](../experiments/implicit-sl/README.md)。

## 按你想改的内容找文件

| 想改什么 | 文件 | 字段 |
|---|---|---|
| 训练三个还是五个模型 | Experiment | `train_seeds`，优先写显式列表 |
| 每个模型训练多久 | Experiment | `train_episodes`；当前研究使用完整 episode |
| 训练哪些故障档位 | Experiment | `training_conditions`、`condition_sampling` |
| 最终测哪个 checkpoint | Experiment | `evaluation.checkpoints` |
| 每档测试几个环境 | Experiment | `test.cases_per_condition` 与对应 `test.cases` 条目数 |
| 学习率、batch size、DDQN探索概率 | Algorithm | `production.policies.<策略名>.learner.params` |
| Base/ISL 的信息、编码或网络 | Algorithm + Python | `observation`、`encoder` / `network` 的工厂与参数 |
| 同角色共享还是各自学习 | Algorithm | 对应策略下的 `parameter_sharing` |
| 订单数量、到达速度、工期或交期 | Scenario | 文件顶部 `jobs` |
| 故障与维修强度 | Scenario | `factory.machines.defaults.breakdown` |
| AGV/机器布局与缓冲区容量 | Scenario / Studio | `factory` 中对应资源和布局 |
| 并发、W&B、资源保护 | Experiment / Profile | `concurrency`、`tracking`、`resources` |

所有相对文件路径都以当前 YAML 所在目录为起点。`target` 是 Python 模块工厂，
不是磁盘文件路径；自定义文件组件使用 `path` / `factory`，按对应组件契约传参。

## 当前研究入口：通常先改顶部

```yaml
train_seeds: [101, 102, 103]
train_episodes: &training_episodes 1000
budget_boundary: complete_episode
checkpoint_every_episodes: 50

training_conditions: [normal, low, medium, high]
condition_sampling: balanced_cycle

final_evaluation: true
evaluation:
  mode: limited
  checkpoints:
    - episode: *training_episodes
      modes: [argmax]
```

`&training_episodes` 定义 YAML 标量锚点，`*training_episodes` 引用该值。
直接编辑文件顶部的训练预算时，最终评估节点随之改变，不用在两个位置改数字。
这是 YAML 读取时的引用；CLI `--set` 在读取后执行，**不会动态重算别名**。
CLI 改预算时还需同步完整的 `evaluation.checkpoints`，或直接编辑文件。

`train_seeds: 3` 也受支持，但含义是三个种子 101/102/103，不是只使用 seed=3；
因此研究文件优先使用显式列表。`train_ticks` 放在高级区域，当前 `train_episodes`
非空时由 episode 预算优先。时间单位均为模拟器物理 tick，`run.seconds` 才是墙钟秒。

## 区分 PPO、Double DQN 与规则

- PPO 的 `epochs` 是遍历当前 rollout 的轮数；Double DQN 是每次更新、每个有样本
  的学习网络从 replay 抽取多少批样本。`batch_size` 都是梯度更新的决策样本数，
  不是 episode 数。
- `clip`、`entropy`、`critic_initial_scale` 对当前 Double DQN 不生效；
  `replay_capacity`、`target_update_interval`、`epsilon` 对当前 PPO 不生效。
  现有公共参数值保留，并在原位置标明适用范围，避免误以为调了就会改变训练。
- `learning_reward_scale` 是 learner 的额外奖励缩放；reward 组件自己的权重另算。
  E1a 的 `reference_jobs`、`reference_ticks` 是归一化参考值，不是订单总数或训练预算。
- Rule 不训练网络，直接评估；不要为它复制学习率或训练种子配置。

角色决策学习策略默认同角色共享。要配置全部独立，可在相应策略下添加：

```yaml
parameter_sharing:
  default: independent
  roles: {}
```

该块位于 `production.policies.dqn` 或相应学习策略名下，与 `learner` 同级。
`roles` 支持按角色覆盖，例如 `roles: {machine: shared}`。
E1a-current 的集中式控制只有一个 central/factory 学习主体，不使用角色共享开关。

## 固定测试清单放在文件末尾

当前同规模 Base/ISL/Central/Rule 的40个测试实例必须一致，训练种子不替换这些种子。
`test.conditions` 是档位清单，`cases_per_condition` 必须等于每档实际条目数，
`seed` 是环境种子，`action_seeds` 是该环境的策略采样种子。

改变订单、地图、故障或物理时间边界，可能会让原 `scenario_sha256` 校验失败：
这是提醒你测试环境变了，而不是格式错误。保留原面板时不要删除或随意改 hash。
有意定义新研究面板时，可以不预设世界 hash，由编译器生成并冻结新世界；此时
不能声称它仍是原测试面板。修改其中一个方法的面板时，应同步同规模的其他对照方法。

## 修改后的检查

在仓库根目录运行：

```sh
uv run smartsom check experiments/implicit-sl/small/base
uv run smartsom check experiments/implicit-sl --json
```

`check` 只读取、编译与校验，不训练、不评估，也不创建正式实验结果。
`'off'` 保持引号，避免 YAML 1.1 将它解释为布尔值；保留质量/时间比例字段原有的
数值字符串引号。列表顺序有语义（规则候选、工艺路线、训练条件等），不要为美观排序。

## 参考与取舍（2026-11-07）

- [PyMARL 默认 YAML](https://github.com/oxwhirl/pymarl/blob/master/src/config/default.yaml)
  使用分段与参数旁注释，区分运行、日志、超参数和 agent 设置。
- [BenchMARL 实验 YAML](https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml)
  就地解释参数单位与 on/off-policy 适用条件，并明确评估和 checkpoint 设置。
- [TorchRL 配置系统](https://docs.pytorch.org/rl/stable/reference/config.html)
  按组件组合配置，支持类型校验和引用；SmartSOM 已有显式 Python 组件与校验，
  本次无需引入 Hydra 或新增配置继承层。
- [Unity ML-Agents 配置说明](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/)
  分开网络、超参数、奖励及算法专属参数，并解释训练步数的计量范围。

本次借鉴这些可读性原则，保留 SmartSOM 的现有字段和所有解析值：常用参数靠前、
科学配置与运行设置分段、注明单位和算法适用范围、长固定清单靠后。
没有将其他项目的参数默认值直接移植到我们的实验。
