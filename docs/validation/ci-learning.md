# Learning CI diagnostics

The learning job installs locked CPU, Gym, RLlib, SB3 and MARL extras. MARL is
required because the full test suite includes resource-PPO checkpoint workflows;
installing only the centralized backends leaves PettingZoo unavailable.

The fixed paired-evaluation test includes training evidence audits, 15 evaluations
and physical/learned replay audits. Its subprocess has a 900-second wall-clock
limit, replacing the 240-second limit exceeded on the September 16 Ubuntu run.
This is an engineering timeout allowance, not a change to seeds, training budgets,
replications, simulation horizons or completion criteria. Ubuntu confirmation
comes from the CI run for the pushed commit, not a local macOS pass.

`scripts/validate_learning.py` emits elapsed times at the start and end of each
training and evaluation audit, in addition to batch progress. The integration
test saves `evaluation-stdout.log` and `evaluation-stderr.log` alongside its
temporary training directories, including when the subprocess times out. Timeout
failures include the latest progress and error output in the pytest log.

The learning job prints its 20 slowest tests. On failure it uploads evaluation
logs, reports and batch summaries as `learning-failure-<attempt>`, retained for
seven days. Use the last started/completed audit and batch counts to distinguish
slow evaluation from an audit or worker that stopped advancing before changing
the time limit again. A timeout remains a failed test.
