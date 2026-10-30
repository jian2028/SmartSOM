# Pre / Post STOP 四组高压力诊断

本批次检验：在相同高压力条件下，Pre、Post 的接纳 STOP 权限分别怎样影响
Base PPO 的学习与调度。它是当前运行流程的单 seed 诊断，不是旧模型的恢复，
也不据此宣称算法优势或定位历史失败的唯一原因。配置准备不代表已经运行。

## 四组对照

| 方法 | Pre accept STOP | Post accept STOP |
| --- | --- | --- |
| both-on | 开 | 开 |
| pre-off | 关 | 开 |
| post-off | 开 | 关 |
| both-off | 关 | 关 |

四份 Algorithm 的科学差异只有上述两个布尔字段。STOP 只终止当前接纳循环，
不等于服务 WAIT，不直接推进物理时间；关闭 STOP 不绕过容量与合法动作检查。
Input 和服务阶段不变，空载 Dispatcher WAIT 保持开启。

## 固定条件与预算

- 地图直接引用公共 `original-20-machine.yaml`，保留原布局、20 台机器和 20 辆 AGV；
  不替换成当前紧凑 Medium 地图。
- 320 个订单，初始 40 单，平均到达间隔 7.2；仅 normal 无故障条件。
  这里的“高压力”是订单到达压力，不是 high 故障条件。
- Base observation + Base encoder + 原生 PPO；所有角色内共享完整网络，
  不跨角色共享。四组使用相同 seed 初始化，规则 Mover 相同。
- 沿用来源 PPO 参数、物理奖励及学习奖励缩放 0.001；不增加 STOP 惩罚。
- 训练 seed 固定为 `[101]`；每组 160,000 累计训练物理 ticks，总计 640,000。
- 精确验证节点为 0 / 80,000 / 160,000；节点 0 是初始化模型。
  cut 截断保留 PPO bootstrap，不能算自然排空。
- 使用最后安全模型做最终测试，不根据测试集选模型；本方案不生成 `best.pt`。
- 保持限时评价和 drain 4000，不在失败后临时放宽上限。
  每条 trial 的 172,800 秒（48 小时）保护上限覆盖训练与评价，不是预计耗时；
  触发上限或面板缺失须报告不完整，不能当作完整四组结果。

同一训练 seed 和 episode 索引对应相同世界生成规则；各组可能在不同时间完成
episode，因此等物理预算不保证消费完全相同数量的训练世界，也不保证优化步数相等。
这些计数应与结果一起报告。

## 案例来源与统计边界

`common/` 的三个文件按字节复制自
`experiments/network-sharing/01-owner-sharing/common/`：
`workload-high.yaml`、`high-validation.json`、`high-test.json`。
固定案例配方带世界指纹，编译时校验并冻结实际世界。

- 验证：602901–602908，sample 动作 seed 603901 / 603902；这些世界已有历史诊断，
  不能称为未见测试数据。
- 测试：604901–604908，sample 动作 seed 605901 / 605902；与验证物理世界分离。
  它是当前仓库新增的独立面板，不冒充历史原始测试集。
- 每个面板分别运行 sample（8 世界 × 2 次）和 argmax（8 世界 × 1 次）。
  每条 trial 三次验证加一次最终测试共 96 个案例运行，四组共 384 个。
- 每组先对同一世界的推理重复汇总，再等权汇总世界；组间比较配对同一世界。
  sample 与 argmax 分开，推理重复不能充当独立训练 seed。

关注交付比例、案例排空率、合格交付覆盖率、拖期、最终合格率和完整成本 J。
未排空的 J 保持缺失；共同排空子集的拖期只能作为有覆盖率标注的补充。
同时比较每组初始化到训练后的变化，不能把动作权限带来的初始差异归因于学习。
单 seed、160k 预算不能证明稳定优劣或收敛；本包不加入 Rule/Random 臂，因此也
不能声称优于规则基线。两个开关可能存在交互，不能假设同时关闭等于两项收益相加。

训练采用汇总采集；验证/测试使用默认详细指标，每 50 物理 ticks 采样。
额外开启有界 decision counters：决策次数按角色/阶段分开，但 STOP 特殊动作
计数目前合计 Pre/Post，不能据此画两者各自的 STOP 概率曲线，也不构成完整因果事件账本。

## 手动启动

在本 worktree 根目录，激活已准备好的环境一次：

```sh
source .venv/bin/activate
smartsom check action-protocol/01-pre-post-stop --profile local --max-concurrent 1
```

`check` 只编译、校验和展示配置，不训练、不连接 W&B。先确认它列出四条 seed-101
任务和预期结果目录。W&B 已登录时不必重复登录；尚未登录可运行 `wandb login`。
凭据不得写进 YAML。

```sh
smartsom run action-protocol/01-pre-post-stop --profile local --foreground --max-concurrent 1
```

四条顺序运行。`local` profile 默认并发为 2，因此这里显式用 CLI 固定为 1；
前台会占用当前终端。W&B 默认 online、project 为 SmartSOM，四条共享 group
`action-protocol-01-pre-post-stop`，各自拥有独立 tracking run 和 variant。
自动基础标签之外添加 diagnostic / pre-post-stop / high-pressure。
网页链接以实际启动输出为准。仅检查配置不验证登录或在线同步。

本分支手动验流程所得属于 development diagnostic。正式研究运行仍须按仓库规则
使用集成 main 或明确 tag；本包不自动提交、打 tag、合并或启动任务。
不要编辑已有运行的冻结输入；要改变科学条件或重做一批，应建立新的具体实验目录。
已有结果不会覆盖，中断后用 `smartsom resume` 和实际 run ID/路径恢复完整结果包。

## 输出、模型与报告

```text
runs/action-protocol/01-pre-post-stop/
├── manifest.json
├── report/
│   ├── report.md
│   ├── metrics.csv
│   └── summary.json
└── trials/
    ├── both-on/seed-101/
    │   ├── last.pt                     # 最新安全模型
    │   ├── continuation/               # 完整续训配套状态，不能丢弃
    │   ├── validation/
    │   │   ├── tick-00000000/model.pt
    │   │   ├── tick-00080000/model.pt
    │   │   └── tick-00160000/model.pt
    │   ├── evaluation/                 # sample / argmax 最终测试
    │   ├── inputs/                     # 冻结配置与案例
    │   │   └── components/builtin/     # 当前实现的源码快照
    │   ├── tracking/
    │   └── progress.json
    ├── pre-off/seed-101/
    ├── post-off/seed-101/
    └── both-off/seed-101/
```

目录按实际运行阶段生成，`last.pt` 存在不等于整批完成。查看状态：

```sh
smartsom runs list
smartsom runs show runs/action-protocol/01-pre-post-stop
```

模型、日志、W&B 本地记录和整批结果均留在被 Git 忽略的 `runs/`，不进入 PR。
PR 只保留可复用的实验配置、案例配方和本说明；临时检查脚本也不提交。
