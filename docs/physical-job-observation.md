# Opt-in physical-job observations

With the learning dependencies installed, use the per-policy built-in selectors
described in [configurable experiments](configurable-experiments.md). Inspection
choice is explicit and different instances can coexist in one process. Default
imports and ordinary policy construction retain the native encoder.

The historical `smartsom.learning.physical_job_observation.install(include_inspection=True)`
entry point remains available for existing process-local integrations. Call it
before preparing or constructing policies and use a fresh process for another
inspection choice. New declarative policies do not require that global install.

The encoder supports the finite-input eight-machine and sixteen-machine layouts (input capacities 10 and 20). It excludes future/external orders and terminal job history, uses physical capacity for global occupancy, and estimates reference work only from released routes and public processing progress. Loaded Dispatcher candidates include eight cargo summary fields; other roles and empty AGVs receive zeros. Eight aggregate job fields are critic-only: PPO actors and native DQN Q scores mask them, while DQN has no critic.

Model metadata carries `physical_job_encoder` with schema `smartsom.physical-job-observation/v1/inspection=True` (or `False`). Older native or local experimental packages are rejected. Fresh initialization is required; never relabel old weights. Bump the schema when feature meanings, normalization or layout change. Git commit and the qualified predecessor hash record implementation provenance separately from compatibility identity.

Legacy installation is process-local and must precede PPO/DQN construction and
package loading. Declarative drivers, learners, spawned samplers and Tune instead
reconstruct the pinned implementation from each policy's frozen selectors. This
module does not create experiments, choose training settings, admit scientific
runs or write checkpoints. Scientific runners, inputs and results remain local.

Continuation encoder state carries the same identity. Session restore validates every driver and sampler policy before restoring simulation, optimizer or RNG state; another inspection choice and legacy identity-free encoder state are rejected. Spawned sampling workers install the frozen observation identity before policy deserialization, for both single-environment sampling and parallel waves. Matching continuation is qualified through fresh native sessions; it does not certify arbitrary in-place RLlib rewinds.

Released rational `reference_ticks` are honored in both original and remaining work estimates; per-machine nominal overrides take precedence, followed by normal-mode time scale and machine rate. Realized latent processing durations remain excluded.
