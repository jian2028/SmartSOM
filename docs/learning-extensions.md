# Public learning extension contracts

Research extensions compose with the existing projection and episode adapters.
They can change observation encoding, feed-forward actor/critic networks and
learning rewards. Semantic action catalogs, legal masks, proposal arbitration,
termination and simulator transitions remain owned by their existing contracts.
Backend training and complete-resume acceptance are separate integration gates;
the environment and registry interfaces below do not establish those gates alone.

## Configuration and registration

`LearningAlgorithm.extensions` is an optional `ExtensionSpec`. `None` is omitted
from serialization, preserving legacy algorithm and checkpoint identities.

```python
from smartsom.config.extensions import (
    ExtensionRef,
    ExtensionSpec,
    NetworkBranch,
    NetworkSpec,
    RewardSpec,
)

extensions = ExtensionSpec(
    observation=ExtensionRef(name="builtin.dict", version="1"),
    network=NetworkSpec(
        actor=NetworkBranch(
            encoder=ExtensionRef(
                name="builtin.mlp",
                version="1",
                parameters={"hidden_sizes": [32], "activation": "relu"},
            ),
            hidden_sizes=(32,),
        ),
        critic=NetworkBranch(hidden_sizes=(64, 32)),
    ),
    reward=RewardSpec(
        team=ExtensionRef(
            name="builtin.reward_scale",
            version="1",
            parameters={"scale": 2.0, "terminal_offset": 0.0},
        ),
        learner_scale=0.01,
    ),
)
```

`role_observations`, `network.roles`, and `reward.roles` accept `machine_policy`
and `agv_policy` overrides. A network override supplies independent actor/critic
branches. Existing resource IDs continue to share their role's parameters.
Role configuration is rejected for centralized providers.

Names resolve only through explicit `register_extension(kind, name, version,
factory, ...)` calls. Configuration cannot import a Python module or execute a
file path. The kinds are `observation`, `reward`, and `torch_encoder`. Registration
declares supported backends and whether an observation/reward component is
stateful. Parameters are finite JSON data.

`bind_extensions(spec, provider)` validates supported backends and pins source
digests. Every factory's Python source is hashed; delegated implementation files
must be included in `source_files`. Built-in Torch factories include the separate
Torch implementation file without importing Torch. Changes after registration or
disagreement with a checkpoint's pinned digest are errors. These digests cover
declared implementation files; framework dependencies have separate identities.

Local workers receive already-authorized Python registration objects through
`export_registrations` and `install_registrations`. Imported models require the
caller to register the same custom implementations explicitly; model configuration
does not automatically execute external extension code.

## Observation and network boundaries

`PublicObservation` contains only a public `DecisionContext`, the existing public
feature vector and named feature groups, plus resource role/agent identity where
applicable. It contains no simulator, complete workload or hidden event calendar.
A custom observation factory accepts `(parameters, ObservationLayout)` and returns
an object with `output_space: ObservationSpace` and `encode(public_observation)`.

Spaces are fixed vectors or fixed-shape numeric Dict fields. Returned values are
checked for exact keys, rectangular shapes and finite values, then converted to
finite float32 by the Gym/PettingZoo boundary. `builtin.vector` preserves the
legacy vector. `builtin.dict` exposes `state` for the central adapter and the
existing `global`, `local`, `candidates` groups for resource agents.

Encoding is separate from semantic mapping. `EncodedDecision` replaces only the
observations delivered to the model; action masks and decoding delegate to the
original immutable projection result. `EncodedResourceDecision` likewise retains
the original joint decision. No extension can replace the mapping through this API.

Each newly requested decision is encoded once. Re-reading an observation returns
the cached encoding. Terminal observations are zeros in the same declared shapes;
legacy central zero masks and resource NOOP-only terminal masks are preserved.

The optional `torch_extensions.build_actor_critic` accepts an `ObservationSpace`,
action count, `NetworkSpec`, backend name and optional role. It builds separate
actor and critic encoder instances and independent feed-forward heads. A custom
Torch factory accepts `(parameters, space)` and returns `nn.Module` with a positive
integer `output_size`. `forward` returns a feature tensor with that final dimension.
Trainable parameters and buffers belong to ordinary model `state_dict`.
Recurrent/stateful Torch encoders are outside this interface and are rejected.

The resulting network's `forward` returns `logits` and `values`; masks remain an
external policy responsibility. `masked_logits` is a convenience for actual action
requests and rejects an empty mask. Value/bootstrap computation can consume a
terminal placeholder without requesting an action.

## Reward accounting

A reward factory accepts `parameters` and returns `transform(transition, reward)`.
`RewardTransition` includes only the before/after public contexts when available,
submitted semantic actions, raw elapsed-time reward, time, termination reason, and
role. It never supplies private simulation state.

The runtime first transforms the team reward once. Resource episodes then apply
each configured role transform once per joint round; agents in that role receive
the same research reward. A stateful team transform is not called once per agent.
Normal, completed, stalled, deadlocked and budget-exhausted transitions all use the
same transform protocol. Initialization failures can have no public context.

`RewardValues(raw, research, learner)` records separate quantities. The learner
value is the research reward multiplied by the positive extension learner scale
and, for resource PPO, its existing `learner_reward_scale`. A backend must apply
that numerical scaling exactly once. Environment `step` returns research rewards;
legacy `step.reward` and `total_reward` retain raw tick semantics. Central steps
also retain `research_reward`, `learner_reward` and encoded-observation digests.
Resource steps retain `RewardBatch(team, roles)` and per-agent encoding digests.
Environment infos expose reward totals. Never sum resource rewards and label that
quantity as the team return.

## State and checkpoint protocol

Stateful observation/reward components must supply JSON-compatible `state_dict()`
and `load_state_dict(state)` methods. Missing hooks reject construction.
`begin_episode()` is an optional lifecycle hook. Stateless registrations explicitly
save `state: null`; undeclared state must not be relied upon by an implementation.
This is an extension-author contract, not a sandbox for arbitrary Python code.

`env.extensions.state_dict()` records implementation identities, parameters,
spaces, numerical scale and component states. A checkpoint with extensions stores
the pinned `extensions` specification and an `extension_state: CheckpointFile`
whose verified file is part of the inference inventory. Network reconstruction
uses checkpoint configuration, not a current preset.

Complete training resume also needs the environment's current encoded observation
cache and accumulated research/learner returns. Use
`env.extension_state_dict()`/`load_extension_state_dict()` for these. Each reset
retains `episode_extension_initial_state` before invoking `begin_episode`.
Reconstruct an active episode from that initial state before resetting and
replaying its recorded actions, then verify/restore the retained current state and
cache. Restoring component state and encoding the current decision again would
incorrectly advance stateful encoders.

Every logical sampling stream owns independent extension state. Saving only one
stream's state cannot establish complete multi-stream resume. An inference export
may explicitly select a stream state; that selection must be recorded separately
from the complete training checkpoint.

## Optional backend network adapters

`ExtensionMaskableActorCriticPolicy` implements SB3's existing masked policy
distribution/value interface, and `ExtensionPPOTorchRLModule` implements RLlib's
forward and `ValueFunctionAPI` interfaces. The framework still owns PPO losses,
optimization and rollout collection. Each receives a JSON `extensions` object
with pinned digests, `provider`, `role` and `fallback_hidden_sizes`: SB3 uses
`policy_kwargs`, RLlib uses `model_config`. Space construction reads the actual
Gym Box or fixed-shape Dict; it never rereads a scenario.

When an extension changes only observations or rewards, the network adapter uses
the recorded fallback hidden sizes and tanh with independent actor/critic paths.
The default `extensions=None` driver path retains the original framework classes.
RLlib permits all-zero masks for terminal bootstrap rows; values never include
the mask. Adapter tests verify actual SB3 PPO updates and model/policy restoration,
plus RLlib module restoration, masks and gradients for both resource roles.
Those tests alone do not establish full driver or multi-stream resume acceptance.

Set `SMARTSOM_REQUIRE_EXTENSIONS=1` in the full optional-dependency quality gate:
missing libraries required by each extension test then fail instead of skipping.
Base-only environments continue to skip only those optional tests.

## Inference transformation evidence

Checkpoint policies expose `on_extension(record)`. An extended run writes these
records to `extension_decisions.jsonl`, once per central model selection or fresh
joint proposal. `smartsom.extension-decision/v1` records the checkpoint, pinned
extension configuration, public context, runtime state before/after encoding,
and every selected view's float32 observation digest, unchanged action-mask
digest and selected index. This is separate from the original joint ledger.

The general run auditor requires this evidence for extended checkpoints. It
restores the retained extension state, replays the recorded selections through
the public checkpoint-policy interface, and compares every record and physical
trace exactly. It does not execute model weights or claim to independently
reproduce stochastic model sampling. Missing, duplicate and altered inputs fail;
an incomplete run receives only a verified-prefix result.
