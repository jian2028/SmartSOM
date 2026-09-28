# 0020 — Composable policies and physical-tick learning

Date: 2026-09-26
Status: Accepted implementation scope; verification is reported separately.

Supersedes ADR 0012's fixed machine/AGV role map, ADR 0013's single algorithm
recipe, and ADR 0017's controlled INTERACT/ranking protocol for new v3 recipes.
ADR 0019's automatic inspection/local disposal and ADR 0018's facility identity
remain in force. Historical configurations, recordings and evidence keep their
original contracts; old weights are never relabeled as v3.

## Ownership

One ProductionSimulator owns inventory, supply reservations, destinations,
service locks, capacity, collisions, inspection and time. Its staged v3 protocol
accepts detached semantic proposals from Machine, Buffer, Dispatcher and Mover.
Policies neither inspect latent truth nor mutate physics. Legacy semantic
commands remain an explicit compatibility boundary in this same kernel.

A source reservation grants a quantity, not a Job or destination inventory.
Freeze eligible supply before this boundary's START. An already-started in-process
job contributes at most one unit when its actual POST has room; new START counts
from the next boundary. Ready inventory and processing supply never double-count.
Pickup ports share the source inventory. Only actual service vehicles participate
in matching. Bind a Job and consume a reservation at loading start; release source
inventory at loading completion. Loading/unloading each consume one physical tick.

Boundary order: due events/supply, Machine/Dispatcher, reservation admission,
service opportunities, Buffer prefix, pickup matching/service, Mover, safety and
one physical commit. Full compatible destinations remain selectable. Non-served
port occupants must leave when an unoccupied adjacent non-port cell is legal.
Conflict rejection preserves the real position; no deadlock rescue is implied.

## Configuration and models

All authoring stays under configs, classified into factories, workloads,
scenarios, policies, compositions, algorithms, rules and runs. A named strategy
group owns shared parameters. Entity overrides bind stable resource identities.
The v3 run owns scenario/composition, backend/algorithm, trainable groups,
physical budgets and independent validation/evaluation inputs. Algorithm profiles
own optimization parameters; policies own rules, new network specifications or
model selectors. References resolve relative to the owning file.

Freeze aliases and archive frozen inference dependencies before execution.
Validation and default final evaluation retain the identical partners/matching.
Best refers to the whole team's validation snapshot. Component export preserves
its source group/update and partner provenance, without asserting standalone
optimality. Resume restores the original combination and all continuation state;
different partners mean a new run. Central PPO is a single whole controller.

## Learning and evidence

Resource RLlib supports PPO and Double DQN; central RLlib/SB3 support PPO.
Count actual physical ticks, decisions, samples and optimizer steps separately.
Keep trajectories separate by environment, episode, owner and policy group.
Buffer uses every candidate and samples only the required prefix. PPO uses one
joint prefix probability and clipping operation per owner/boundary. DQN retains
prefix states, masks, actual interval rewards and gamma**dt, including dt=0.
Save replay, target/exploration clocks and random streams explicitly.

Old weights require retraining. Factory compatibility is required. Optional
frameworks stay outside the core and import boundary. No formal research launch,
performance claim, commit or push is authorized by engineering verification.
