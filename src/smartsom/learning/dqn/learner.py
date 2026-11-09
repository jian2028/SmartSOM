"""Native masked Double DQN on complete per-resource decision transitions.

Gamma is one, as specified by the Base world objective. Rewards between decisions
include all intervening physical ticks, including rule-controlled decisions.
An unresolved tail at a safety cut is not a terminal training sample. It is
counted and omitted; no unobserved next action or terminal value is invented.
"""

import copy
import random

import torch
from torch import nn

from smartsom.domain.observation_data import FEATURES, SET_NAMES
from smartsom.learning.dqn.replay import PackedRows
from smartsom.learning.heads import CandidateHead
from smartsom.learning.models import EncodedModel, make_models, packer
from smartsom.learning.ownership import NetworkOwnership
from smartsom.learning.social_actor import Pool, SocialActor
from smartsom.learning.subjects import groups
from smartsom.representations.base.batching import pack
from smartsom.representations.base.features import BaseEncoder

_NATIVE_MODULES = nn.Module.modules
_NATIVE_NAMED_MODULES = nn.Module.named_modules


class BaseDQN:
    needs_next_request = True

    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        torch.set_num_threads(numerical_threads)
        torch.manual_seed(seed)
        self.options, self.training, self.selection = options, training, selection
        # Match the deterministic native forward contract (gradients still run
        # during update); target inference must never mutate running statistics.
        self.groups = groups(options)
        self.models = make_models(options, value_head=False).eval()
        self.ownership = NetworkOwnership(options)
        self.models = self.ownership.expand(self.models)
        self.pack = packer(options)
        self.targets = copy.deepcopy(self.models).eval()
        for parameter in self.targets.parameters():
            parameter.requires_grad = False
        self._pending_target_init = {
            k
            for k, m in self.models.items()
            if any(
                isinstance(v, torch.nn.parameter.UninitializedParameter)
                for v in m.parameters()
            )
        }
        self.optimizers = {
            key: torch.optim.Adam(
                self.models[key].parameters(),
                lr=self.ownership.options[g].learning_rate,
            )
            for g in self.groups
            for key in self.ownership.keys[g]
        }
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.packed_replay = (
            options.network is None
            and options.encoder.target
            in {
                "smartsom.representations.base.encoder.create_encoder",
                "smartsom.representations.social.encoder.create_encoder",
            }
            and options.observation.target
            in {
                "smartsom.representations.base.observation.create_observation",
                "smartsom.representations.social.observation.create_observation",
            }
        )
        storage = PackedRows if self.packed_replay else list
        self.records, self.replay = storage(), {key: storage() for key in self.models}
        from smartsom.learning.update_schedule import UpdateClock

        self.update_clocks = {key: UpdateClock() for key in self.models}
        self.key_updates = {key: 0 for key in self.models}
        self.updates = 0
        self.decisions = 0
        self.truncated_tails = 0
        self.subject_transition_counts = {}
        self.last_losses = {}
        self.bound_policy_states = {}

    @property
    def decoder(self):
        return "epsilon-greedy" if self.selection == "sample" else "greedy-q"

    def needs_critic(self, request):
        return False

    def _pack(self, requests):
        return self.pack(
            requests,
            None,
            self.options.count_scale,
            allow_empty_wait=self.options.allow_empty_wait,
        )

    def _singleton_evaluation(self, model):
        """Only stateless native networks may omit an unused Q value."""
        if (
            type(self) is not BaseDQN
            or self.training
            or self.selection != "argmax"
            or self.options.network is not None
            or self.options.encoder.target
            not in {
                "smartsom.representations.base.encoder.create_encoder",
                "smartsom.representations.social.encoder.create_encoder",
            }
            or self.options.observation.target
            not in {
                "smartsom.representations.base.observation.create_observation",
                "smartsom.representations.social.observation.create_observation",
            }
            or self.pack is not pack
            or "_pack" in self.__dict__
            or type(model) is not EncodedModel
            or type(model.encoder) is not BaseEncoder
            or type(model.head) is not CandidateHead
            or "actor" in model.__dict__
        ):
            return False
        # Hooks and replaced native submodules can have observable effects even
        # for one candidate. Do not suppress their normal forward execution.
        from torch.nn.modules import module

        if module._global_forward_hooks or module._global_forward_pre_hooks:
            return False
        native = (
            EncodedModel,
            BaseEncoder,
            CandidateHead,
            nn.ModuleList,
            nn.ModuleDict,
            SocialActor,
            Pool,
            nn.Sequential,
            nn.Linear,
            nn.Tanh,
        )

        def original_check():
            return all(
                type(layer) in native
                and not layer._forward_hooks
                and not layer._forward_pre_hooks
                and "forward" not in layer.__dict__
                for layer in model.modules()
            )

        if (
            nn.Module.modules is not _NATIVE_MODULES
            or nn.Module.named_modules is not _NATIVE_NAMED_MODULES
        ):
            return original_check()
        # Inspect the live graph without recursive named_modules generators.
        # Do not cache hook or child identities across requests.
        pending, seen, checked_types = [model], set(), set()
        while pending:
            layer = pending.pop()
            if layer is None or layer in seen:
                continue
            seen.add(layer)
            kind = type(layer)
            if kind not in native:
                return original_check()
            if kind not in checked_types:
                if (
                    kind.modules is not _NATIVE_MODULES
                    or kind.named_modules is not _NATIVE_NAMED_MODULES
                ):
                    return original_check()
                checked_types.add(kind)
            if (
                "modules" in layer.__dict__
                or "named_modules" in layer.__dict__
                or type(layer._modules) is not dict
            ):
                return original_check()
            if (
                layer._forward_hooks
                or layer._forward_pre_hooks
                or "forward" in layer.__dict__
            ):
                return False
            if layer._modules:
                pending.extend(reversed(tuple(layer._modules.values())))
        return True

    def choose(self, request, critic=None):
        model = self.models[self.ownership.key(request)]
        batch = None
        if len(request.candidates) == 1 and self._singleton_evaluation(model):
            if self.options.social_arm == "base" and model.head.social is None:
                return request.candidates[0].action
            # Keep packing validation even for roles without a social head.
            batch = self._pack([request])
            weight = model.encoder.candidate[0].weight
            values = [
                batch["context"],
                batch["candidates"],
                *(v for v, _ in batch["sets"]),
            ]
            native_shape = (
                batch["context"].shape == (1, FEATURES)
                and batch["candidates"].shape == (1, 1, FEATURES)
                and len(batch["sets"]) == len(SET_NAMES)
                and all(
                    v.ndim == 3 and v.shape[-1] == FEATURES for v, _ in batch["sets"]
                )
                and all(
                    v.dtype == weight.dtype and v.device == weight.device
                    for v in values
                )
            )
            # Malformed shapes/dtypes follow the original forward and its errors.
            if native_shape:
                social = model.head.social
                if social is not None:
                    active = int(batch["stage"][0]) in social.stages
                    if active != bool(batch["social_active"][0]):
                        raise ValueError(
                            "social observation does not match enabled role/stage"
                        )
                return request.candidates[0].action
        with torch.no_grad():
            q = model.actor(batch if batch is not None else self._pack([request]))[0]
            key = self.ownership.key(request)
            if key in self._pending_target_init:
                self.targets[key].load_state_dict(model.state_dict())
                self._pending_target_init.remove(key)
        explore = self.selection == "sample" and (
            float(torch.rand((), generator=self.generator))
            < self.update_clocks[self.ownership.key(request)].epsilon(
                self.ownership.options[request.group]
            )
        )
        index = (
            int(torch.randint(len(request.candidates), (), generator=self.generator))
            if explore
            else int(q.argmax())
        )
        if self.training:
            self.records.append({"request": request, "index": index})
            self.decisions += 1
        candidate = request.candidates[index]
        return (
            candidate.identity
            if self.options.control == "base"
            and not self.options.action_representation.merge_aliases
            else candidate.action
        )

    def update(self, rewards, *, terminal, final_critic=None, episode_end=True):
        from smartsom.learning.diagnostics import UpdateDiagnostics
        from smartsom.learning.dqn.replay import bounded_transitions, transitions

        scales = {g: s.learning_reward_scale for g, s in self.ownership.options.items()}
        result = (bounded_transitions if self.packed_replay else transitions)(
            self.records,
            rewards,
            terminal=terminal,
            scale=scales,
            gamma={g: s.gamma for g, s in self.ownership.options.items()},
            split_stages=self.options.learning_policy.split_stages,
            subject_counts=self.subject_transition_counts,
            **({"ownership": self.ownership} if self.packed_replay else {}),
        )
        pending, omitted = result[:2]
        self.truncated_tails += omitted
        self.discard_rollout()
        losses = {}
        diagnostics = UpdateDiagnostics()
        for group in self.groups:
            settings = self.ownership.options[group]
            for key in self.ownership.keys[group]:
                if self.ownership.is_frozen(key):
                    continue
                replay = self.replay[key]
                if self.packed_replay:
                    new_count = result[2][key]
                    replay.merge_newest(pending[key], settings.replay_capacity)
                else:
                    new = [
                        r
                        for r in pending[group]
                        if self.ownership.key(r["request"]) == key
                    ]
                    new_count = len(new)
                    replay.extend(reversed(new))
                    del replay[: -settings.replay_capacity]
                updates = self.update_clocks[key].advance(
                    new_count, settings, episode_end=episode_end
                )
                if not replay:
                    continue
                for _ in range(updates):
                    rows = self.rng.sample(
                        replay, min(len(replay), settings.batch_size)
                    )
                    inputs = self._pack([r["request"] for r in rows])
                    q = self.models[key].actor(inputs)
                    selected = torch.tensor([r["index"] for r in rows])
                    predictions = q.gather(1, selected[:, None]).squeeze(1)
                    target = torch.tensor(
                        [r["reward"] for r in rows], dtype=torch.float32
                    )
                    nonterminal = [
                        i for i, r in enumerate(rows) if r["next"] is not None
                    ]
                    if nonterminal:
                        inputs_next = self._pack([rows[i]["next"] for i in nonterminal])
                        with torch.no_grad():
                            action = self.models[key].actor(inputs_next).argmax(1)
                            value = (
                                self.targets[key]
                                .actor(inputs_next)
                                .gather(1, action[:, None])
                                .squeeze(1)
                            )
                        target[nonterminal] += value * torch.tensor(
                            [
                                settings.gamma ** rows[i]["physical_dt"]
                                for i in nonterminal
                            ]
                        )
                    loss = nn.functional.smooth_l1_loss(predictions, target)
                    with torch.no_grad():
                        diagnostics.add(
                            rows,
                            {
                                "q_loss": nn.functional.smooth_l1_loss(
                                    predictions, target, reduction="none"
                                ).tolist(),
                                "absolute_td_error": (target - predictions)
                                .abs()
                                .tolist(),
                                "q_value": predictions.tolist(),
                            },
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite Base DQN loss")
                    optimizer = self.optimizers[key]
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.models[key].parameters(), 1.0)
                    optimizer.step()
                    self.updates += 1
                    self.key_updates[key] += 1
                    # Shared roles retain the established global update clock.
                    # Independent owners use their own update count and interval.
                    for shared in self.groups:
                        interval = self.ownership.options[shared].target_update_interval
                        if (
                            self.ownership.keys[shared] == (shared,)
                            and self.updates % interval == 0
                        ):
                            self.targets[shared].load_state_dict(
                                self.models[shared].state_dict()
                            )
                    if (
                        key != group
                        and self.key_updates[key] % settings.target_update_interval == 0
                    ):
                        self.targets[key].load_state_dict(self.models[key].state_dict())
                    losses[group] = {
                        "q": float(loss.detach()),
                        "replay_size": len(replay),
                        "transitions": new_count,
                    }
            if group in losses:
                losses[group]["update_diagnostics"] = diagnostics.summary(group)
        for subject in self.subject_transition_counts:
            losses.setdefault(subject[0], {})
        for group, values in losses.items():
            settings = self.ownership.options[group]
            values["subject_transitions"] = [
                {
                    "subject": list(key),
                    **counts,
                    "omitted_fraction": counts["omitted"] / counts["observed"],
                }
                for key, counts in sorted(self.subject_transition_counts.items())
                if key[0] == group
            ]
            values["owner_epsilon"] = {
                key: self.update_clocks[key].epsilon(settings)
                for key in self.ownership.keys[group]
            }
            values["epsilon"] = sum(values["owner_epsilon"].values()) / len(
                values["owner_epsilon"]
            )
            values["owner_target_updates"] = {
                key: (self.updates if key == group else self.key_updates[key])
                // settings.target_update_interval
                for key in self.ownership.keys[group]
            }
            values["target_updates"] = sum(values["owner_target_updates"].values())
            values["replay_size"] = sum(
                len(self.replay[key]) for key in self.ownership.keys[group]
            )
        self.last_losses = losses
        return losses

    def get_state(self):
        from smartsom.learning.dqn.replay import CheckpointReplay

        # Custom observations can carry tensors or other extension payloads;
        # retain Torch's normal serialization/map-location handling for those.
        builtin = self.options.observation.target in {
            "smartsom.representations.base.observation.create_observation",
            "smartsom.representations.social.observation.create_observation",
        }
        return {
            "ownership": self.ownership.contract,
            "key_updates": self.key_updates,
            "update_clocks": {k: c.state() for k, c in self.update_clocks.items()},
            "models": self.models.state_dict(),
            "targets": self.targets.state_dict(),
            "optimizers": {g: o.state_dict() for g, o in self.optimizers.items()},
            "replay": CheckpointReplay(self.replay) if builtin else self.replay,
            "updates": self.updates,
            "decisions": self.decisions,
            "truncated_tails": self.truncated_tails,
            "subject_transition_counts": self.subject_transition_counts,
            "rng": self.rng.getstate(),
            "generator": self.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

    def set_state(self, payload):
        self.ownership.validate(payload["ownership"])
        from smartsom.learning.update_schedule import UpdateClock

        self.update_clocks = {
            key: UpdateClock.restore(state)
            for key, state in payload["update_clocks"].items()
        }
        self.key_updates = payload["key_updates"]
        self.models.load_state_dict(payload["models"])
        self._pending_target_init = {
            k
            for k, m in self.models.items()
            if any(
                isinstance(v, torch.nn.parameter.UninitializedParameter)
                for v in m.parameters()
            )
        }
        self.targets.load_state_dict(payload["targets"])
        self.models.eval()
        self.targets.eval()
        for group, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][group])
        self.replay = payload["replay"]
        if self.packed_replay:
            self.replay = {key: PackedRows(rows) for key, rows in self.replay.items()}
        self.updates = payload["updates"]
        self.decisions = payload["decisions"]
        self.truncated_tails = payload["truncated_tails"]
        self.subject_transition_counts = payload.get("subject_transition_counts", {})
        self.rng.setstate(payload["rng"])
        self.generator.set_state(payload["generator"])
        torch.set_rng_state(payload["torch_rng"])

    def save(self, path):
        from smartsom.learning.state import save

        save(path, self.options, self.get_state())

    def load(self, path):
        from smartsom.learning.state import load

        self.set_state(load(path, self.options))

    def sampling_state(self):
        return {
            "ownership": self.ownership.contract,
            "models": self.models.state_dict(),
            "update_clocks": {k: c.state() for k, c in self.update_clocks.items()},
        }

    def load_sampling_state(self, state):
        from smartsom.learning.update_schedule import UpdateClock

        self.update_clocks = {
            key: UpdateClock.restore(value)
            for key, value in state["update_clocks"].items()
        }
        self.ownership.validate(state["ownership"])
        self.models.load_state_dict(state["models"])
        self._pending_target_init = {
            k
            for k, m in self.models.items()
            if any(
                isinstance(v, torch.nn.parameter.UninitializedParameter)
                for v in m.parameters()
            )
        }

    def seed_sampling(self, seed):
        self.generator.manual_seed(seed)

    def discard_rollout(self):
        self.records = PackedRows() if self.packed_replay else []
