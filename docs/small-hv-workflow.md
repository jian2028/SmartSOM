# Small H/V：配置与运行

当前主入口为 `check/run --task train-evaluate --study PREPARED_DIRECTORY`，
停止、恢复与查看分别用 `stop`、`resume`、`monitor`，见
[统一入口](command-workflow.md)。本页保留原 worktree 阶段的说明和兼容命令，
以便追溯验证记录；已有冻结 Study 仍须通过源码身份检查。

本轮入口是 `configs/test/studies/small_hv.yaml`。Small 来自 Template 7：19×9，
8 台机器、8 辆 AGV、4 种工序。共 24 组：G0/G1 × Low/Mid/High V ×
PPO/DQN × 自动矩阵/全零矩阵。Social Learning 没有加入。

代码在独立 worktree，尚未 commit。下面命令直接使用已安装环境，不进行依赖同步。
完整试跑由你启动；实现验证中的小预算训练不属于这 24 组。

## 哪个文件控制什么

| 要修改什么 | 编辑位置 |
|---|---|
| 地图、端口、容量、同质机器 | `configs/test/factories/small.yaml` |
| G1 速度和质量 | `configs/test/factories/small_h1.yaml` |
| 工艺目录、个性化、到达槽位和 V | `configs/test/workloads/small_content.yaml` |
| 自动最短行程时间及覆盖 | `configs/test/transport/small_auto.yaml` |
| 瞬移 | `configs/test/transport/small_zero.yaml`，显式 `default_ticks: 0` |
| 自己填写行程时间 | `configs/test/transport/small_manual_example.yaml` |
| 各角色网络、规则或 checkpoint | `configs/test/policies/`，由组合文件引用 |
| 角色共享组与资源覆盖 | `configs/test/compositions/small_train_ppo.yaml` 或 `small_train_dqn.yaml` |
| 配货规则 | 组合的 `pickup_matching` 引用 `configs/test/rules/` 小文件 |
| PPO/DQN 优化参数 | `configs/test/algorithms/ppo.yaml`、`dqn.yaml` |
| 批量条件、预算、三套 seed、验证次数和 episode 时限 | `configs/test/studies/small_hv.yaml` |
| 单场景的模式和引用 | `configs/test/scenarios/small_matrix_auto.yaml` / `small_matrix_zero.yaml` 是四订单例子 |

常规使用只编辑上述配置。需要改实现时，新增代码按职责放在这些位置，
没有整体搬目录：

| 代码（相对于 `src/smartsom/`） | 职责 |
|---|---|
| `config/travel_time.py` | 矩阵文件校验、相对路径与 Factory 最短路生成 |
| `domain/travel_time.py` | 不依赖学习框架的矩阵和物理兼容契约 |
| `engine/production_protocol.py` | 物理核内的行程、到达队列、预约及服务阶段 |
| `domain/factory_design.py`、`engine/production.py` | 机器速率倍率、统一时长取整和有限订单终止 |
| `workloads/job_content.py` | 内容池、DP 距离、精确耦合、历史更新和 V 分档 |
| `experiments/composable_study.py` | 冻结 24 组、进程并行执行、恢复、参照和结果汇总 |
| `experiments/composable.py` | 沿用统一训练/评估、模型产物和继续状态 |
| `api.py`、`experiments/cli.py` | Python API 和命令行入口 |
| `studio/replay_inspector.py` | 矩阵行程、剩余时间和排队状态的回放展示 |

Small 的 normal/fast 缺陷概率采用 MD 的 3%/5%，并非旧模板的 1.8%/3%。
H 用固定机器速率倍率及质量表实现，不随机改 Job 时长。跨 H 保持地图与完整兼容关系。

内容配置每池 64 件，16 窗×4 件，120 秒/窗，空厂启动。delta 控制原工序
时长变化，alpha 控制合法单道插入。V 按最终有序工艺及参考时长计算，先与历史
比较再更新；它不是随机到达强度。三档仅重排同池单件订单，保留槽位、规格、
总工作量和每件宽裕量。准备结果给出各输入的实际 V、H 及取整诊断。

## 从准备到运行

```sh
cd /Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM
source .venv/bin/activate

# 冻结数据与配置，不训练
smartsom study prepare --config configs/test/studies/small_hv.yaml --output runs/prepared/small_hv_parallel

# 查看 24 组、实际 H/V、矩阵身份、预算和状态
smartsom study show runs/prepared/small_hv_parallel

# 看一组的绑定、训练/冻结状态、模型身份及随机种子
smartsom show-config --config runs/prepared/small_hv_parallel/config/runs/h0_low_ppo_auto.yaml

# 整批训练、validation、测试与配对参照；默认最多 8 个独立实验同时运行
smartsom study run runs/prepared/small_hv_parallel

# 中断后恢复，不重复已完成组
smartsom study resume runs/prepared/small_hv_parallel

# 显式重试失败组
smartsom study resume runs/prepared/small_hv_parallel --retry-failed
```

同一路径的重复 prepare 在原配置、全部引用文件、模型引用与源身份不变时返回已有准备。
修改源配置、代码或依赖后使用新的输出目录，例如 `runs/prepared/small_hv_parallel_v2`。
准备失败的目录保留诊断，不自动删除或覆盖。
显式重试会重新评估失败的初始对照，并把原失败结果保存在 `controls/*.failed-*.json`。

产物目录中的 `config/` 是展开后的分类 YAML，`datasets/` 是冻结 Job 池与
V 曲线，`snapshots/` 是批量执行实际使用的准备快照。`experiments/` 保存各子实验，
`controls/` 保存配对参照，`summary.json` / `summary.csv` 汇总结果。
**修改展开后的 YAML 不会修改冻结批量任务**；需要修改源文件并重新 prepare。

## 进度面板修复后重新开始

2026-09-26 的显示修复使用固定高度的 Rich 独立终端屏幕，一轮收集全部 worker
后统一刷新，普通进度至多每秒一次，状态未变化时不重绘。训练 ticks 与当前
validation/evaluation case 的 ticks 分别显示；没有确定计数的阶段不播放假进度。
退出时恢复原终端并保留最终文字摘要。终端较小时会注明未显示的 active 任务数量。

旧 `small_hv_parallel` 已停止并保留 checkpoint。源码身份发生变化，因此使用
下面的新准备目录从头开始，不修改旧冻结实验来规避身份检查：

```sh
cd /Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM
source .venv/bin/activate
smartsom study prepare --config configs/test/studies/small_hv.yaml --output runs/prepared/small_hv_parallel_stable
smartsom study run runs/prepared/small_hv_parallel_stable
```

已准备好时只需运行最后一行。默认仍是 8 个独立实验并行，每个实验 1 个采样环境。
之后正常中断可用 `smartsom study resume runs/prepared/small_hv_parallel_stable`。
不需要另开 `monitor` 才能查看进度。训练/验证预算、种子及模型配置没有修改。

## 总进度、ETA 与表格面板

2026-09-27 的面板把总体流程进度与 ETA 放在顶部。下面每行是一个实验，列出
算法、H/V、矩阵模式、当前阶段、训练 ticks、已结束案例数和当前案例 ticks。
不再为每个实验重复 PID、执行方式和优化计数；详细字段保留在日志与进度 JSON。
训练进度在 validation 时暂停，当前案例计数仍更新。

总进度按计划预算加权，包含训练、validation、最终测试、初始模型与共享
rule/random 对照；它不是已经执行的物理 tick 总数。ETA 至少采集 30 秒推进数据
后显示近似值，使用最近最多五分钟的实际墙钟推进速率。启动时显示 `estimating`。
失败、成功、结束、运行与排队数量分开报告，进度到 100% 不代表模型表现合格。

上次 `small_hv_parallel_stable` 已正常停止，保留八个 1,024-tick checkpoint。
这次源码再次变化，重新开始使用新目录；数据、三套种子、预算与 8 并行设置相同：

```sh
cd /Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM
source .venv/bin/activate
smartsom study prepare --config configs/test/studies/small_hv.yaml --output runs/prepared/small_hv_dashboard
smartsom study run runs/prepared/small_hv_dashboard
```

准备完成后只需最后一行。以后恢复用
`smartsom study resume runs/prepared/small_hv_dashboard`。不需要另开 monitor。

## 只跑一组，或把 train 和 evaluate 分开

```sh
# 训练 + 每次 validation + 最终独立测试
smartsom train-evaluate --config runs/prepared/small_hv_parallel/config/runs/h0_low_ppo_auto.yaml

# 只训练 DQN，稍后测试
smartsom train --config runs/prepared/small_hv_parallel/config/runs/h0_low_dqn_auto.yaml
smartsom evaluate RUN_DIRECTORY --checkpoint last

# 指定组合的 validation 最佳快照
smartsom evaluate RUN_DIRECTORY --checkpoint best

# 恢复一个训练任务
smartsom resume RUN_DIRECTORY

# 查看已有回放
smartsom playback EVALUATION_CASE_DIRECTORY
```

`RUN_DIRECTORY` 换成训练命令打印的实际目录。单独运行原冻结配置后，整批入口
会按身份采用兼容的训练结果。若更换伙伴、数据或优化参数，会形成另一实验。
`best` 若未形成合格快照会明确报错，不伪装成 `last`。

训练入口默认更新 Machine、Buffer、Dispatcher 三组。同组共享权重，各组独立。
矩阵 Mover 绑定 `automatic_travel`，没有可学习的移动动作。它不是冻结的模型，
也不制造训练样本。中央 PPO 与网格 Mover 的原有入口保留。

## 自己填写或覆盖矩阵

```yaml
schema: smartsom.travel-time-matrix/v1
source: auto
overrides:
  input_pickup_top:       # 示例名：必须替换为本 Factory 的实际 port_id
    machine_pre_top: 7
```

节点是实际 `port_id` 及 `initial:<agv_id>`；准备快照的场景中保存完整节点与矩阵。
当前可编辑的 `small_manual_example.yaml` 已使用真实端口 ID。需要核对全部节点时：

```sh
python - <<'PYTHON'
from smartsom import api
p = api.prepare(api.load_config("configs/test/runs/small_rules_auto.yaml"), training=False)
for point, x, y in p.scenario.transport_matrix.points:
    print(point, x, y)
PYTHON
```

`auto` 未覆盖的条目从 Factory BFS 得到。`manual` 不计算网格距离：每个有向条目
必须填写，或明确给出 `default_ticks`。对角线为零，整数单位是一个物理 tick；
`null` 表示不可达。每个可能取货口必须能到达该 Job 下一合法阶段，
否则启动前报错，避免 Buffer 选出无法配货的前缀。手填可非对称；默认值不能隐式补缺。

```yaml
schema: smartsom.travel-time-matrix/v1
source: manual
default_ticks: 0
overrides: {}
```

这就是瞬移条件。装卸仍各一 tick，检查与报废仍耗时，PRE/POST/Inspection
容量及端口服务互斥保留。派车只预约来源数量，**不预留目的地空位**。
正行程途中不改派；到达满位后排队，也可重新选择合法目的地。
矩阵模式没有道路占位或碰撞，因此不能把其结果直接当作真实网格交通表现。

## 预算、结果与模型组合

每组训练 16,384 ticks，更新间隔 256 ticks；每 4 次更新验证 5 个固定独立案例。
生成配置通过 **5 个 scenarios × 每场景 1 replication** 表达五个不同 Job 池，
而非对同一个显式池重复五次。最后测试另有五个独立池。

24 组训练合计 393,216 ticks，另有 1,920 个 validation episode、120 个模型测试
以及初始/规则/合法随机参照。验证、测试和参照不计入训练预算。
默认最终测试 `last`，`best` 按完整组合的 validation 选择。

汇总的 `V_train` 是训练输入的实测 V，`V_test` 是五个测试池的均值，
`V_test_by_case` 保存逐案例数值，避免用训练 V 代替测试输入的 V。

记录交付、奖励、flow time、waiting、makespan、截断和异常；矩阵结果另含
机器利用率、平均 WIP、Job 等待、行程及端口等待的 tick 计数。
原有 waiting 是预约等待与目的地容量等待之和；`job_waiting_ticks` 则累计全部
活跃 Job 的非加工、非检查、非运输和非装卸等待，二者不能混作同一个指标。

规则参照使用 Machine normal-first/SPT、Buffer EDD、最近合法目的地，
随机参照选择合法动作，二者使用相同矩阵。初始模型在训练前保存，最后才在
测试输入上评估，测试结果不参与 checkpoint 选择。

各角色仍可在独立 `evaluate --config` 中引用不同实验、不同 checkpoint 和不同
兼容网络。参考 [组合模型工作流](composable-workflow.md)。本轮组件记录运输与
取整契约和来源矩阵；网格/矩阵契约不兼容时明确拒绝。矩阵内更换数值属于
显式新评估条件，记录变化；resume 则恢复原矩阵与原伙伴。

首轮为未提交源码的开发试跑，单训练 seed，不宣称学习效果优于规则。
正式研究仍需在按仓库要求冻结的集成提交或明确 tag 上重新准备。

最终检查和证据边界见 [Small 工程验收记录](small-hv-verification.md)。


## 默认并行方案与终端状态

`configs/test/studies/small_hv.yaml` 的 `max_concurrent: 8` 控制同时运行的独立实验数。
可改为 6 或 4；改后重新 prepare 到新目录。各实验内部保持 `num_envs: 1`、
`sampling_processes: 0`、`numerical_threads: 1`、`device: cpu`；5 个验证案例逐个执行。
这是旧 PPO 性能测试中最快的并行方式；该测试不证明新版 PPO/DQN 的加速比。
不通过减少验证案例、tick 时限或更改 seed 来加速。

终端显示实际活动/排队/结束实验数、PPO/DQN、H/V、行程模式、训练 ticks 百分比、
真实优化次数、各组决策/样本/优化次数、validation/evaluation 的案例数及当前案例 ticks。
`last activity` 是子进程上报时间；父进程刷新不等于子进程有新进展。
达到训练 ticks 预算后，实验还需要保存、评估和参照；只有全部完成才计入结束。
每个独立实验训练 Machine/Buffer/Dispatcher，Mover 为 automatic_travel 规则。

父进程是 study.json 和总进度的唯一写入者；每个子进程写自己的 workers/<entry>/
日志与状态。共享 rule/random 参照用文件锁计算一次。Ctrl+C 会向活动子进程
转发中断，并保留已完成更新的 recovery checkpoint。resume 不重复已完成实验。

另开终端只读监视：

```sh
source .venv/bin/activate
smartsom monitor runs/prepared/small_hv_parallel
```

旧 `runs/prepared/small_hv` 保留此次串行中断的证据，不改写它的冻结身份。
代码改变后不能直接 resume 旧来源；新入口从新的冻结组合开始训练。

## 统一阶段面板与本次重新运行

新版顶部是总流程进度、Elapsed 和 ETA，表头汇总 Factory、地图大小、算法、H/V、
Social Information、三类 seed 和实际并行设置。宽终端用两列实例卡片，窄终端用紧凑
表格；当前阶段用短条，更新和保存没有确定百分比时显示已用时间。
采样 ⇄ 更新显示已完成轮次及距下次验证的轮数；本配置仍是 64 轮、每 4 轮验证，
16 批 × 5 个固定 validation 案例，最后 5 个独立 evaluation 案例。

在 `configs/test/studies/small_hv.yaml` 的 `logging.title` 修改总标题；
`logging.task_title` 修改实例标题模板。支持 `{algorithm}`、`{H}`、`{V}`、
`{travel}`、`{seed}`、`{name}`、`{id}`。当前 training seed 仍只有 101；
面板能显示不同 seed 的独立实例，但本次没有增加多 seed 的实验编排或预算。

新源码身份使用新准备目录，旧数据与 checkpoint 不覆盖。无需额外 monitor：

```sh
cd /Users/jianni/.codex/worktrees/composable-policies-v3/SmartSOM
source .venv/bin/activate
smartsom study prepare --config configs/test/studies/small_hv.yaml --output runs/prepared/small_hv_workflow
smartsom study run runs/prepared/small_hv_workflow
```

以后在源码保持不变时恢复用 `smartsom study resume runs/prepared/small_hv_workflow`。
默认并行仍是 8，每个实验 1 个环境、CPU、1 个数值线程，内部顺序采样/验证。
`train`、`train-evaluate`、独立 `evaluate` 使用相同布局，按各自流程省略不适用阶段。
可通过 `--progress-title "我的标题"` 临时覆盖总标题，不改冻结配置或产物目录。
