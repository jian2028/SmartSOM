# 中图 shared Base / SL：归因与学习诊断

本包位于 `experiments/v2-diagnostics`，与 `experiments/v2` 同级。研究步骤和预定判读见 [研究计划](研究计划.yaml)。
这是开发诊断，不覆盖旧训练进度，也不把不同目标的结果当成算法复现。

## 第一阶段已实现的官方入口

| 组 | 原始截断归因 | 有限任务窗口归因 |
| --- | --- | --- |
| Base shared | check-base-original / long-base-original | check-base-window / long-base-window |
| SL shared | check-sl-original / long-sl-original | check-sl-window / long-sl-window |

短验证每组2集，节点0/1/2用于测试完整生命周期；长实验每组1000集，节点0/50/.../1000。
所有节点固定同一12场（四条件各3），节点0使用未训练模型；最终测试是另一个80场面板。
除了原始/window learner工厂，两臂配置一致。Base与SL只改变social观测/网络。
不固定HPOF或normal、不加压力观测、不改变shared所有权、原生replay后端、reward或优化器。
先前 independent 的诊断属于另一组对照，不混合统计。

## 终止语义

- all_delivered：自然完成，两个learner均terminal。
- drain_deadline：物理指标仍为截断；original继续原生尾部遗漏；window明确把它作为有限任务窗口的学习终点，尾部回报写入、bootstrap为0。
- training_budget_cut / wall_clock_interrupted：不得伪装为成功或任务终点，仍按原安全截断处理。
- BLOCKED：只是资源状态，不等于经过证明的吸收死锁。故障维修、未来到货、合法搬运仍可能改变状态。

window是显式目标消融，不是声称原实现违反契约。两组均未新增剩余窗口时间输入，因此保留部分可观测限制。
审计按主体统计，team奖励区间相互重叠，不可把各主体tail_return相加称为全局丢失奖励。
源码仅给RewardTimeline添加结束原因；组件通过官方path/factory加载，不替换官方调度器。

## 启动

在本分支 checkout 根目录激活环境后：

```bash
smartsom check experiments/v2-diagnostics/中图/归因与学习
smartsom run experiments/v2-diagnostics/中图/归因与学习 \
  --id check-base-original --id check-base-window \
  --id check-sl-original --id check-sl-window \
  --max-concurrent 4 --evaluation-processes 1 --evaluation-cpus 1 \
  --cpus 16 --memory-gib 64 --wandb offline \
  --output-root /home/jianni/runs/v2-credit-learning-check-20261108 --background
```

短验证和审查通过后，四个id替换为long-*，输出使用新的long目录。
不要对已有根目录重复启动，也不要用其他checkout运行这些新配置。
运行中通过官方configure调整并发，科学变量不在续跑中修改。

## 后续阶段

学习曲线阶段与第一阶段同时采集。质量目标及SL预算阶段的完整变量、门槛与指标见研究计划，按先后证据推进，不提前选择质量阈值或假定窗口方案必然更好。
这些后续阶段尚未生成可误启动的experiment.yaml；达到门槛后固定选定目标与组件、生成官方入口再启动。
保留独立test，禁止用test选择权重。之前结果见 [已有checkpoint诊断](../历史诊断/README.md)。

官方并行case worker目前拒绝path形式自定义组件；本组evaluation_processes=1，四实验并行。不能把请求并发数高于1当作已支持。

新配方短验证stage10、长实验stage20，与既有stage1–3避免并发上限冲突；当前已启动短验证使用此前冻结的stage1元数据，其科学设置相同。归档全部配置不表示应一次启动整个主题。

## 完成短验证后自动进入长实验

`scripts/after_short.py` 只读取官方status，核对四臂的2次训练更新、0/1/2三组固定面板和边界审计，然后调用官方run启动四个long入口。它不调度worker或执行仿真；失败/暂停时停止，已有长实验输出时拒绝重复启动。

```bash
python experiments/v2-diagnostics/中图/归因与学习/scripts/after_short.py \
  --short /home/jianni/runs/v2-credit-learning-check-serial-20261108 \
  --long /home/jianni/runs/v2-credit-learning-long-20261108
```

`--once` 只核查一次，不会启动长实验。源码/配置一致性、恢复测试和性能审查是启动此脚本前的独立工程门槛；脚本通过不代表window已经改善学习。
生成物在 `../results/` 统一查看；该目录是指向服务器运行存储的本地链接且被Git忽略，克隆分支不会下载模型或日志。


## 长实验的来源隔离

服务器短验证四臂均完成，训练边界与0/1/2固定评估面板通过检查；但该批次从
未提交的集成工作区启动，运行期间提交使Git HEAD从306cca2变成新提交，
批次收尾的严格来源检查失败。冻结与当前implementation_sha256、Python和
依赖版本一致；这不能把旧批次整体失败状态改写成成功。

保留旧批次及其错误记录，核验独立trial证据后，以本分支已提交版本创建专用
运行副本，使用官方run在新输出目录启动四个long入口。运行副本及其独立环境
在实验结束前不pull、不切换提交、不修改源码或同步依赖；后续开发继续在分支
工作区进行。不要关闭来源校验、修改旧manifest或从新提交强行resume旧批次。
新运行manifest记录实际启动提交。此前after_short进程已因partial退出，不再
复用该自动门槛；此次启动依据已核验的四个独立trial证据。实验科学设置不变。
