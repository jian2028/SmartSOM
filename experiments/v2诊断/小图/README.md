# v2诊断

本包使用官方 `smartsom check/run/pause/resume`，不替换训练循环。
基线源码为 5d76d5b；本工作树为 codex/v2-diagnostics。
只运行 Mac Small；保留 Job、buffer、breakdown、reward 与 Mover 配置。
学习主体全部 independent，优先 seed 101。历史实验和清理自动化不变。

## 三个阶段

1. **学习链路**：先用真实完整 episode 审计每个主体最后一个动作、
   尾部 reward、是否写入 replay；先不改变 terminal 语义。
   若需固定任务窗口目标，新增显式配置并与原语义作同预算长训练对照。
2. **状态可辨识性**：当前候选已包含来源类型、占用率、阻塞和承诺车辆。
   对照候选局部状态与额外的全局缓冲压力上下文；不重复增加现有字段。
3. **疏通规则**：对照学得策略、始终 HPOF、仅机器阻塞时优先疏通；
   不改物理合法性、不把启发式称为死锁保证。

每阶段先审计/短验证、分析，再决定长验证；计划每阶段一个 1000 episode
的验证预算，包含匹配对照，具体 arm 在该阶段分析后冻结。
已有测试面板仅作历史诊断；新增独立验证面板用于方案选择，最终评估另设。
所有阶段报告 overdue/order、完整完工条件下 mean tardiness、完成率、
机器 processing/blocked/starved 时间、质量、reward 和 tick/min。
未完成订单不按 tardiness=0；单 seed 结果不代表跨 seed 优势。

## 当前执行入口

`stage1-audit` 是 8 个完整训练 episode 的审计，不是长训练结论。
组件只增加 update diagnostics，不改变 replay、动作、奖励或终止语义。
官方入口会冻结配置及组件源码、保存恢复点。

```
smartsom check experiments/v2诊断/小图/stage1-audit
smartsom run experiments/v2诊断/小图/stage1-audit --wandb disabled
```

## 已扩展的官方能力

- RewardTimeline.end_reason 提供结束原因；原生默认行为不变。
  create_window_learner 仅将 drain_deadline 作为学习终止，保留尾部样本。
  这是目标语义对照，不是宣称原安全截断实现违反了既有约定。
- continuation_keep_generations 默认 null；新配方设为 2，官方 commit
  在校验当前恢复代与 last 后保留当前及前代。不触及 best、周期评估副本。
  未发布的不完整/歧义目录保留并记录；trash 支持中断恢复。
  原定时清理保持暂停，旧实验不启用新配置。

## 运行顺序与分析门槛

不要直接 run 整个 topic：它包含备用组合与短检查配方。
stage1-check 已执行 original/window 各8轮；stage1-long 为两者各1000轮。
后续按 stage1 的独立验证结果选择 original 或 window 前缀，再运行
stage2-long 中相应 pressure 变体。stage3-long 选择前阶段方案的 blocked
和 hpof 变体。未经选中的备用组合不运行。

pressure 同时增加输入维度和改变该编码层初始化；结论仅限整个观测方案，
不能把单seed差异完全归因于新增信息。blocked是启发式，无死锁保证。
hpof从头训练时固定pickup为HPOF（不可用时NPF）、其他主体仍学习，不等同于历史checkpoint干预。

panels/validation.json 与 test.json 使用新的独立种子域并锁定物理hash；
32个验证案例用于250/500/1000轮分析，80个test仅在方案冻结后使用。
测试和验证与seed101前1000个训练场景均已独立核验无物理hash交叉。
比较使用最后1000轮模型，不根据holdout选checkpoint；验证不足以支持
改变方案时保留原语义，并把无改善或不稳定结果如实记录。

第一阶段两组1000轮长诊断已通过官方入口启动；尚未完成，不得把启动当作结论。
工程证据见 analysis/engineering.md。
当前只做Base DDQN、independent、seed101诊断；不据此宣称ISL或跨seed结果。

HPOF基线开局没有POST时必须回退NPF；这是明确的复合规则。旧无回退配方的失败运行保留，后续使用新冻结输入。第一阶段运行中的original/window组件未改变。
