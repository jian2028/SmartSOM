# 0059 — Explicit algorithm components and action permissions

Status: implemented; local engineering acceptance and review are recorded in
[algorithm components validation](../validation/algorithm-components.md). Hosted CI
remains a separate merge gate.

Supersedes the algorithm declaration and extension-loading portions of ADR0058.
The one physical core, production semantics, tracking defaults and historical
source boundary remain in force. This change does not introduce a second simulator.

Algorithm schema `smartsom.algorithm-components/v1` separates control mode, decision protocol,
production policy bindings and traffic. Pre/post acceptance STOP permissions are
independent. Empty Dispatcher pickup WAIT remains a separate existing permission.
Defaults retain both STOP permissions and disable empty WAIT. No service WAIT or
per-entity overrides are introduced. Centralized control rejects distributed
action declarations while retaining shared reward and scaling parameters.

Production declares one default and explicit role.stage bindings to named policy
instances. At most one learning instance is allowed; rule/random/custom policies
can handle other decisions. Traffic binds its own joint Mover and is not changed
by choosing a production baseline. Candidates retain semantic identities and core
legality is authoritative.

PPO and masked Double DQN expose explicit factories, separate update/data handling
and reuse applicable feature encoders. Base/social observation and encoder
contracts are explicit; the independent critic and social special-action scoring
remain unchanged. A complete network factory is an alternative to an encoder.
Built-in Base observations reuse protocol-prepared local fields without copying
full public-factory state. Custom projections receive detached public facts and
can replace only observation inputs, never the legal candidate collection.

Module targets identify packaged SmartSOM factories. External file factories can
declare a source_root; local dependencies then use relative package imports.
Compilation freezes the declared source bundle and records its fingerprint,
including resources. Single files cannot silently import adjacent local files.
Third-party libraries remain environment dependencies. No general discovery or
third-party trainer compatibility layer is added.

The learner contract provides decision records, updates, sampler-state transfer,
explicit sampling seeds and complete training state. Updates keep the existing
complete-episode transaction boundary; interrupted incomplete work restarts from
the last durable commit. Components, action permissions and bindings are science
identity, never operational resume overrides. Checkpoint schema v2 rejects older
component layouts even where a tensor happens to have the same shape.

Only an explicit migration command reads the former algorithm schema. It writes
new inputs and a report, never edits originals or migrates checkpoints. Old runs
remain linked to their original source, environment and evidence. Formal research
runs, new waiting semantics and centralized DRL remain outside this change.
