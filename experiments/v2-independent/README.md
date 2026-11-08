# V2 independent-parameter Base/ISL campaign

Each of Small, Medium and Large has six recipes: Base/ISL × training seeds
101, 102, 103. Names include `independent`; shared counterparts and Morris/Justin
comparators are in [v2-shared](../v2-shared/README.md).

The independent algorithms set `parameter_sharing.default: independent` with no
role overrides. Each physical Machine, PRE, POST and Input has its own network,
target, optimizer and replay. Dispatcher pickup/delivery remain separate subjects
per AGV. Accept/service heads within the same buffer still belong to that buffer.
Network owners total 45/111/221 for Small/Medium/Large. No shared checkpoint is
converted or reused as an independent training checkpoint.

Scenarios, job distributions, rewards, network architectures, learning parameters,
1000-episode budgets, four-condition training mix and 80-case evaluation panels
match the shared counterparts. Independent ownership increases the number of
updates and aggregate replay capacity; equal episodes do not imply equal compute.
Tracking groups and entry names distinguish ownership. Stages 1/2/3 prioritize
seed 101, then 102 and 103. Morris is absent from this independent queue.

Use `smartsom check experiments/v2-independent/small` before a new launch, and
select one scale explicitly. Mac runs Small; BudgetVM runs Medium. Large is
configured and validated only, not authorized for formal training in this campaign.
Never launch this parent directory as a combined three-scale batch.

Previously launched runs retain their frozen source and input paths. The existing
stage-0 additions under `runs/v2-expanded-20261108` are an operational queue with
seed-101 entries submitted first; changing these tracked recipes does not rewrite
that live queue or restart its work. Runtime concurrency is controlled separately
with official `batch-limit`/`batch-budget`; resource guards remain active.

Regenerate both families with `uv run python scripts/build_v2_experiments.py`.
For physical definitions, censored metrics and provisional checkpoint selection
limitations see the shared campaign documentation and each run's recorded evidence.
