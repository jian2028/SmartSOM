# 0062 — Human scenario authoring and Studio defaults

Status: implemented; engineering verification and PR CI are separate evidence.

Supersedes the four-file authoring boundary of ADR0058/0061 and the static-only
current editing boundary of ADR0014/0015. Historical records remain unchanged.

Daily inputs are Scenario, Algorithm and Experiment. A scenario combines factory
geometry, device defaults/overrides, job generation and machine fault rules.
Each machine explicitly names its semantic ID, single operation and grid position.
Machines and AGVs occupy exactly one cell. Machine size and rotation are not scene
fields; imports with larger machines fail instead of silently changing geometry.
Buffers, ports and other existing physical objects retain their explicit spatial
contracts. AGV heading remains an initial-state parameter, not geometric rotation.

Machine and AGV parameters inherit document-local defaults. Machine Pre/Post
buffers inherit role defaults. Explicit overrides, including a value equal to its
default, remain pinned. Mapping fields merge; lists replace; changing a storage
discriminator replaces that storage definition. Missing fields inherit, unknown
fields fail. No template inheritance or environment-dependent defaults are added.

The job editor exposes the existing generation law: finite arrivals, uniform route
choice, optional repeated first operation, processing variation, urgency and due
date multipliers. Fault intervals/repairs retain the existing conditional truncated
exponential draws and per-machine random streams. The normal condition disables
faults; low/medium/high rules live with machine defaults and may be overridden per
machine. Experiment chooses training/evaluation conditions. The A/B/C/D and
single-capability current execution boundary remains unchanged.

One compiler expands scenes into the existing typed factory/workload contracts.
Expansion runs before sampling, not in physical ticks. Execution freezes the
human scene in inputs/authoring/scenario.yaml and fully resolved physical inputs
alongside it. Science identity uses resolved inputs; raw input checksums remain
provenance. Editing a scene never migrates an old checkpoint. Historical evidence
and models retain their original source/identity requirements.

Studio New offers current Small, Medium and blank scenes. Common settings, entity
overrides and job generation are editable forms with transaction-based undo/redo.
Save/recovery preserve authoring structure. Case preview uses an explicit seed,
fault condition and drain horizon and does not run policies or connect tracking.
Raw factory imports cannot reconstruct an old workload: use explicit experiment
migration to preserve it. Legacy factory reliability uses a different, previously
unconnected protocol and is rejected rather than silently reinterpreted.

Expanded four-file inputs remain the internal execution/frozen-input format.
Existing explicit four-file callers remain readable through that same parser;
there is no second physical runtime. Config migrate can create a scene experiment
from a current four-file experiment without overwriting sources or destinations.
Historical geometry/rendering helpers remain for evidence tests and playback,
not as selectable new-scene templates.
