# 0075 — Algorithm-neutral scenarios and inline study test panels

Status: implemented; verification is recorded separately.

Supersedes ADR0073's four-entry/current-panel organization. The user authorized
Small/Medium × Base/ISL/E1a-current PPO/Rule, with inline Experiment test seeds and
no separate authored Case files. Core physics, actor inputs, Base/ISL algorithms,
training seed domains and the existing 80 test worlds remain unchanged.

One machine-breakdown Scenario per scale holds geometry, jobs and all fault
parameters. normal generates no outages; low/medium/high select the distribution.
Experiment selects Algorithm, conditions, seeds, budget and inline test recipes.
Each test recipe retains case ID, environment seed, action seeds and world hash.
The compiler validates cardinality and world identities, then freezes generated
case inventories in run inputs. Runtime inventory files remain artifacts, not a
fourth manually authored scientific input. Inline completion cases use the effective
calibrated protection horizon; controlled recovery with inline test panels is
explicitly rejected because it constructs its own intervention worlds.
Existing low-level frozen inventory
reading and other callers remain supported; no old run files are migrated.

Learning entries each use 101/102/103 and 1000 complete episodes with a balanced
four-condition cycle. Final tests contain ten cases per condition. Rule has one
evaluation task per scale and no trainable models. This yields 18 model tasks and
two Rule tasks (800 final evaluation episodes), not separate models per condition.
Central is E1a-current PPO with its own reward; cross-method physical metrics
remain distinct from differently defined learning returns.

This changes configuration/source identities. Existing checkpoint continuation
uses its frozen source. No run, commit, push or formal research claim is made.
