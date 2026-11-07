# SmartSOM roadmap

The current direction is one physical simulator, two decision mechanisms and one
component interface. [ADR0073](decisions/0073-unified-observation-components.md)
records the observation refactor and historical-source retirement.

## Current implementation and acceptance

The simulator models production, finite buffers, grid AGVs, failures and quality.
Reservation decisions and centralized direct assignments share physical services.
Scenario/Algorithm/Experiment authoring, source-frozen execution, current replay,
Studio, monitoring and complete-update recovery use the current runtime.

Native PPO/DDQN support Base and implicit SL with shared/independent role ownership.
Morris is an adapted global-network comparator on reservation decisions. Rule,
Random, Justin and custom components remain available. Centralized direct-task PPO/Double DQN, custom critic and reward components are
extended under [ADR0074](decisions/0074-custom-training-components.md).

The active acceptance work is unified observation/Rule interfaces, native-only
initialization, cleanup of old authored interfaces, independent code review and
paired performance tests with no representative-case slowdown above 5%. Passing
engineering tests does not establish convergence or scientific superiority.
Hosted Linux/macOS CI remains distinct from local evidence.

## Subsequent research work

Observation simplification is a subsequent controlled change: define Base fields
and additional SL fields in their own Python components, select parameters in
YAML, and compare with the preserved current observation. It is not silently
included in this structural refactor.

Formal campaigns start from integrated main or an explicit tag, recording source,
worlds, seeds, ownership, initialization and evaluation protocol. Historical
milestone/results documents are source-version evidence in the external archive;
they do not expose alternative current runtimes.
