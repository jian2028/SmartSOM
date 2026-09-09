# 实验操作指南

本入口把配置、训练、评估与证据目录连接起来。底层 PPO 仍由 RLlib 或
SB3 执行；模拟器、动作、合法性、NOOP 和协调规则维持原有契约。
当前实施与验收状态见 [实施记录](implementation-usability.md)，正式证据见
[validation](validation/)。历史第 12/13 项脚本继续使用冻结配方。

## 安装和检查

```sh
uv sync --locked --extra learning --extra cpu --extra reports --extra tensorboard
uv run --no-sync smartsom doctor --preset marl_micro
```

`learning` 安装三条路线；精简安装可以换成 `learning-marl`、`learning-rllib`
或 `learning-sb3`。`cp`、`tensorboard`、`reports`、`wandb` 分别选择。
只装基础模拟器时执行 `uv sync --locked`。没有选中的可选依赖不算 doctor 错误。
选了 TensorBoard 但没有安装时，启动前会明确失败。

`doctor` 默认只读，列出解释器、实际导入路径、源码提交、已安装依赖和配置。
`--probe` 显式执行设备计算检查。`uv run --no-sync` 使用项目环境而不重新同步；
也可以 `source .venv/bin/activate` 后直接执行下文 `smartsom`。

CPU 默认运行。CUDA 必须在配置中明确选择；不可用时失败，不自动回退。
`cpu` 和 `cuda` extra 互斥，二者保持 Torch 2.14。macOS 使用 PyPI 的相同版本，
Linux 的显式 profile 使用 PyTorch CPU 或 cu130 索引。锁解析不等于 Linux/CUDA
实测，后者本轮待验证。依赖分流依据 [uv 官方方式](https://docs.astral.sh/uv/guides/integration/pytorch/)。

## 配置与预览

```sh
smartsom presets list
smartsom presets show marl_micro
smartsom show-config --preset marl_micro --steps 4096 --num-envs 4
```

优先级是预设 → 用户文件 → CLI 显式覆盖或 Python 属性修改。不同入口产生等价
科学配置时，科学输入摘要相同；来源记录、实验名称和显示开关不改变科学身份。
CLI 重复设置同一字段（包括 `--steps` 与 `--set training.total_steps=...`）会报错。
未知字段、错误类型和预算不整除也在启动前失败。

用户文件可仅覆盖需要的部分，相对路径以该文件所在目录为基准：

```yaml
preset: marl_micro
training:
  total_steps: 4096
algorithm:
  learning_rate: 0.0003
runtime:
  num_envs: 1
  device: cpu
output:
  name: baseline
  root: ../runs
```

```sh
smartsom show-config --config experiment.yaml
smartsom train --config experiment.yaml --seed 101
```

`scenario`、`algorithm.source` 可以指向自己的场景和算法文件。
`scenario_overrides` 对现有场景做严格校验的局部覆盖；例如
`--set scenario_overrides.arrivals.profile.initial_job_count=1`。
生成分布和模块约束仍由场景模型验证。

## 训练、验证与保存

```sh
smartsom train --preset marl_micro --name baseline --seed 101 --steps 4096 \
  --steps-per-update 256 --num-envs 1 --device cpu \
  --set algorithm.learning_rate=0.0003
```

`training.total_steps` 是全体环境的采样预算。资源 MARL 一步是一个联合轮，
不是一个物理动作，也不是单个 agent step。`training.steps_per_update` 是一次
PPO 更新的总采样量；预览显示每环境配额。后端内部优化次数另行记录。

`--num-envs 4 --sampling-processes 2` 用两个采样进程执行四个逻辑环境；
`--sampling-processes 0` 在主进程执行这些环境。每次更新对各环境使用相同配额，
按固定环境 ID 汇总，由一个证据写入者落盘。`--threads` 控制数值线程数；
批量训练的 `--max-concurrent` 另行控制独立训练器数量，三者不是同一个参数。
多环境使用明确版本的 episode ID 和种子分配；原单环境配方保持原序列。

默认每 4 次更新做一次验证：seed303、5 个固定输入、独立环境和随机状态。
默认验证不做完整回放，独立评估开启回放。验证输入基于既有场景结构，
更换 seed 不自动构成跨场景泛化证据。

默认每 4 次更新保存一次，保留最近 2 份，正常结束保存 `last`。
`best` 按完工率优先；相同完工率且完成集合一致才比较 makespan。
相同完工率但完成集合不一致时保留已有 best 并记录覆盖差异。
全部未完工不会产生 best。`validation.best_mode=all_complete` 只接受全完工；
`custom` 需要明确指标、方向和失败处理。`last` 保存不依赖 best。

常用字段：

| 配置组 | 字段 |
| --- | --- |
| `training` | `total_steps`, `steps_per_update`, `max_decisions`, `max_ticks` |
| `algorithm` | `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `entropy_coefficient`, `batch_size`, `n_epochs`, `hidden_sizes`, `learner_reward_scale` |
| `runtime` | `num_envs`, `sampling_processes`, `numerical_threads`, `max_concurrent`, `device` |
| `validation` | `enabled`, `every_updates`, `seed`, `replications`, `deterministic`, `full_replay`, `best_mode`, `metric`, `direction`, `failure_policy`, `patience`, `min_delta` |
| `checkpointing` | `every_updates`（`null` 关闭周期保存）, `keep_last`, `save_last`, `save_best` |
| `output` | `name`, `root`, `tags` |

提前停止默认关闭（`validation.patience=null`）；它与搜索剪枝是不同设置。
第一次 Ctrl+C 请求在下一完整更新边界保存并退出；第二次允许立即中止。
中断和提前停止有独立状态，不表示原定预算完成。

## 评估、续训和初始化

```sh
smartsom evaluate RUN_DIRECTORY --checkpoint last --seed 202 --replications 5
smartsom evaluate RUN_DIRECTORY --checkpoint best --baseline spt
smartsom train-evaluate --preset marl_micro
smartsom resume RUN_DIRECTORY
smartsom train --preset marl_micro --initialize-from MODEL_PACKAGE
```

评估默认只运行选定模型，基线通过 `--baseline` 明确加入。输入可以是运行目录、
checkpoint 或导出的模型 ZIP；读取元数据识别后端与结构。不同基础结构需满足模型
兼容约束，否则明确拒绝。`--scenario` 可指定测试场景；报告记录实际输入和覆盖。
`--no-deterministic` 使用独立算法随机流采样，`--no-replay` 关闭完整评估回放。

评估的 `input_coverage` 同时记录工厂／作业是否与训练一致、实际环境输入摘要，
以及与已记录训练 episode、固定验证集的重叠。缺少历史账本时标为不可用；
checkpoint 的账本可能尚未包含活动 episode，因此无已知重叠不等于训练样本完全不相交。
评估引用的新版 update 会登记保护引用，自动保留策略不能删除它；旧 checkpoint
的原始文件不改动。

`resume` 保留原实验预算、优化器、PPO 动态状态、RNG 和活动 episode。
只有相同源码、科学配置、采样拓扑、设备类型与锁定依赖才可完整恢复。
执行中的环境从固定输入与已记录动作前缀重建；MARL 也包含 NOOP 和拒绝。
恢复不重复计步或学习，先前 attempt 和 checkpoint 保留。

`initialize-from` 只使用已有权重开始新实验，重建 optimizer、随机状态、计数和身份。
旧 model-only checkpoint 可评估和初始化，不能冒充完整恢复点。

## 日志与证据

`--progress auto/on/off` 控制进度条。`--verbose 0/1/2` 分别显示摘要、周期指标、
详细诊断。`--log-format json` 适合批处理；非交互输出没有动态终端控制序列。
本地结构化事件始终保留，显示开关不改变训练随机状态。

Quickstart 开启 TensorBoard；关闭用 `--set logging.tensorboard=false`。
开启后执行 `tensorboard --logdir runs` 查看曲线。
W&B 默认关闭，需 `--extra wandb`、`logging.wandb=true` 和
`logging.wandb_project=PROJECT`；离线用 `logging.wandb_mode=offline`。
本次验收只测试离线，不向云端发布。

每次实验只有一个真实目录，内部登记训练、评估与恢复阶段：

```text
runs/<date>-<name>-<id>/
  run.json
  config/
  logs/
  checkpoints/
  evaluation/
  evidence/
  reports/
```

`smartsom runs list` 查看实验，`runs show` 查看单个实验，`index rebuild` 在真实
实验之外重建 models/logs/reports 快捷入口。删快捷入口不会删实验。
报告、静态图与 ZIP 的详细用法见 [报告与导出](reporting.md)。

## 历史配置和场景创建

```sh
smartsom migrate configs/runs/learning_marl.yaml --output my_training.json
smartsom init generated_fjsp my_fjsp
smartsom import-fjs example.fjs --output-dir imported --instance-id example
```

迁移保留原配方预算、场景和算法，并关闭历史配方没有的周期验证与 TensorBoard。
原文件不改写。原 `run`、`train` 的 v1 输入仍经兼容层读取；历史正式验收脚本
直接调用原执行 API，保持原第 12/13 项证据语义。
