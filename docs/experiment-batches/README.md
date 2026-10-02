# SmartSOM experiment batches

Each batch has a frozen plan, a separate result record, and links to its local evidence. A plan or a launched run is not a measured result. Generated runs remain outside Git.

| Batch | Question | Plan | Result | Evidence state |
| --- | --- | --- | --- | --- |
| 01 · Small H/V | How do PPO and DQN behave across machine heterogeneity, content variation, and travel time? | [Study contract](../small-hv-workflow.md) | [Batch 01](batch-01-small-hv.md) | Completed one-seed development pilot |
| 02 · Base diagnosis | Does Dispatcher shaping repair missed pickups under auto transport? | [Batch 02 plan](batch-02-plan.md) | [Batch 02 results](batch-02-results.md) | Development evidence: calibration-off parent stopped after six training entries; two fresh supplements completed training with 1/5 and 0/5 held-out cases complete; historical r4 evaluation remains failed |

The next research stage, Social Learning, is not part of Batch 02. It should use a base configuration selected from the measured outcomes rather than from a planned reward change.
