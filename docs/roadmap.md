# SmartSOM roadmap

The current direction is one physical simulator, two decision mechanisms and one
component interface. [ADR0073](decisions/0073-unified-observation-components.md)
records the observation refactor and historical-source retirement.

## Current implementation and acceptance

The simulator models production, finite buffers, grid AGVs, failures and quality.
Reservation decisions and centralized direct assignments share physical services.
Scenario/Algorithm/Experiment authoring, source-frozen execution, historical replay reading,
Studio, monitoring and official iteration recovery use the current runtime.

Official RLlib PPO/DQN and Tune replace native training for Base and implicit SL.
The user-approved [core closeout](decisions/0098-rllib-core-closeout.md) covers
training, fixed panels, official best/last, calibration, whole-run control and
basic Optuna. ADR0100 adds Rule/Random fixed-panel execution; formal comparator
campaigns are not launched. Learning standalone evaluation, window/recovery analysis,
new recording remains deferred. PR42 Morris/E1a-current PPO is being integrated
under ADR0101; its engineering acceptance is separate from formal experiments.

Representative three-pair PPO/DQN performance evidence meets the five-percent
median steady/end-to-end gate. Local Linux owner checks and independent core
review passed; these remain separate from scientific acceptance. Existing native regressions and historical experiments
do not establish new framework behavior. Engineering tests do not establish
convergence or scientific superiority; hosted/platform evidence remains separate.

## Subsequent research work

Observation simplification is a subsequent controlled change: define Base fields
and additional SL fields in their own Python components, select parameters in
YAML, and compare with the preserved current observation. It is not silently
included in this structural refactor.

Formal campaigns start from integrated main or an explicit tag, recording source,
worlds, seeds, ownership, initialization and evaluation protocol. Historical
milestone/results documents are source-version evidence in the external archive;
they do not expose alternative current runtimes.
