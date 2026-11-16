# 0091 — Official evaluation policy also selects best

Status: user authorized; integrated migration acceptance remains incomplete.

This supersedes the fixed-argmax best selection and the explicit sampled-DQN
epsilon override in ADR 0089. The user requested official evaluation defaults
and explicitly confirmed that PPO best uses the sampled panel, without a second
argmax panel.

For pinned Ray 2.58, DQN evaluation defaults to `explore=False`; PPO evaluation
inherits `explore=True`. Use the resolved official evaluation configuration and
record its selection policy in the frozen panel and results. Do not add an
evaluation epsilon override or a second policy-selection panel. Training remains
unchanged. Fixed world, condition and action seeds are still part of the panel.

Best minimizes mean J across every prescribed case in the complete periodic
panel under that policy, including all prescribed action-seed replicates.
Any incomplete/truncated case makes the whole panel ineligible. There is no
complete-case-only mean. Final test cases are not used for model selection or
Optuna objectives. The three Optuna objectives retain their separate meanings.

Historical frozen runs and ADR 0089 remain intact. New configuration using a
conflicting explicit case mode must receive a migration error, not silently
lose cases. Engineering tests and this decision do not establish the full
runtime migration or the required long-training performance acceptance.
