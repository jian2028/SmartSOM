# 分级设计依据

这是针对用户成本与质量偏好的本地执行方案。L0/L1/L2 及 0/1/2 名 reviewer 是本 skill 的设计，不是下列机构规定的行业标准，也没有经本项目实验测得 token 节省比例。

## 借鉴的公开做法

- [Google：Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)：以独立、可理解的小改动降低审查负担；大小不是单纯行数。用于约束审查范围，不能据此声称 Google 允许自审替代其人工审批。
- [Google：What to look for](https://google.github.io/eng-practices/review/reviewer/looking-for.html)：审查设计、功能、复杂度、测试和上下文；复杂并发等问题需要对应能力。用于按实际风险分配职责。
- [CodeRabbit：review / full review](https://kb.coderabbit.ai/articles/1442026547-troubleshoot-why-coderabbit-reviews-might-not-trigger)：公开区分增量审查和完整审查。用于修复后只复核增量，不能据此推断其 agent 数或内部 token 预算。
- [Anthropic：多智能体研究系统](https://www.anthropic.com/research/multi-agent-research-system)：强调投入随复杂度变化、职责明确、避免重复，并指出多智能体的 token 成本。其研究系统成本观察不直接外推为本 code review 的成本比例。

## 分级边界示例

| 改动 | 预期选择与原因 |
|---|---|
| 修正文档错别字/失效链接，无行为说明变化 | L0；链接与差异检查，不运行仿真 |
| 局部变量重命名 | 确认无动态引用或接口影响后 L0；否则 L1 |
| 新增一个普通 CLI 参数的解析和范围校验 | L1；单 reviewer 覆盖入口与拒绝行为 |
| 一行 YAML 改 reward 权重、训练/测试种子域或截断语义 | L2；科研契约影响不能按行数降级 |
| 修改内存保护比例、资源预留重算、失败任务选择性 retry | L2；覆盖配置到调度/恢复的实际调用链与未选任务保持性 |
| 修改仿真热路径缓存/共享状态 | L2；语义保持、失效与适用的配对性能测量 |
| 修复 L1 审查发现的一处边界问题 | 原 reviewer 做增量复核，不另派整组 reviewer |
| 高风险改动，但当前对话禁止子代理 | 保留 L2 判断，执行允许的自查/测试，标明独立审查缺口，不委派 |

普通改动不会因属于 SmartSOM 或文件中出现科研名词自动升为深入审查。小补丁也不会因只有一行而免除对高风险契约的检查。
