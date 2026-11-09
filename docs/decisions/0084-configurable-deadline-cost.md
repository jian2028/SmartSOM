# 0084 — Configurable task-deadline costs and learning boundaries

Date: 2026-11-09. Status: user-authorized.

Amends ADR0083's unchanged reward and bootstrap choice for the V3 campaign.
The existing completion reward accepts optional `truncation_fixed_penalty` and
`truncation_unfinished_weight`, both defaulting to zero. At an actual drain
deadline with unfinished demands, charge B + lambda * U/N once, in normalized
reward units (outside the ordinary cost scale). N is the original demand count;
U excludes all delivered demands, including failed-quality shipments. Quality
failure remains a separate cost. No successful-completion bonus is introduced.
The reward transition carries the classified `end_reason`; a physical tick
limit shortened by the training budget is not a task deadline.

Native learners expose `drain_deadline_terminal: false` by default. Opting in
makes only an actual drain deadline a learning terminal: DQN retains the final
observed decision-to-boundary return and has no next-state bootstrap, while PPO
uses zero boundary value. Budget/wall interruptions preserve their existing
continuation/omission semantics. Physical status remains truncated, not completed.
This is a finite-window objective with unchanged partially observed stationary
policies, not a claim of a fully observed finite-horizon state representation.

V3 explicitly enables these general capabilities in its algorithm YAML files:
B=10, lambda=10, drain_deadline_terminal=true, gamma=1, reward scale=1. The usual
cost remains -(dt + overdue_delta/N + 1000*failed_delta/N)/1000. These penalty
coefficients are user-selected initial settings, not empirically calibrated optima.
All seven algorithm configurations share this reward definition, including Rule,
Random and authored-only Central. Large remains authored-only. Incomplete final
metrics and learning_cost stay censored; learning_return includes the penalty.

No network, replay schema, optimizer, observation, job distribution or physical
rules change. Changed scientific inputs require a newly frozen run; do not resume
old campaign checkpoints under the new objective. Other experiments retain the
default-disabled behavior unless they explicitly opt in.
