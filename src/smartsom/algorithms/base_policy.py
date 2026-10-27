"""Startup-bound Base policies over detached semantic decisions.

Policy bindings select behavior; the existing PPO object continues to own its
five parameter groups, optimizers and physical-time return accounting.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

from smartsom.algorithms.base_rules import BaseRule

DECISION_POINTS = frozenset(
    {
        "machine.process",
        "dispatcher.pickup",
        "dispatcher.delivery",
        "pre.accept",
        "pre.service",
        "post.accept",
        "post.service",
        "input.accept",
        "input.service",
    }
)
BUILTINS = frozenset({"rule", "random", "ppo"})


def effective_policy_names(options):
    return {options.bindings.get(point, options.method) for point in DECISION_POINTS}


def uses_learning(options):
    return "ppo" in effective_policy_names(options)


def _provider(spec, seed):
    path = Path(spec.path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise ValueError(f"policy provider source mismatch: {path}")
    module_spec = importlib.util.spec_from_file_location(
        "smartsom_base_provider_" + spec.sha256, path
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load policy provider: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    policy = getattr(module, spec.factory)(seed=seed)
    for name in ("choose", "get_state", "set_state"):
        if not callable(getattr(policy, name, None)):
            raise ValueError(f"policy provider requires {name}(): {path}")
    return policy


class BoundBasePolicy:
    """Route decisions without importing or encoding nonlearning choices per step."""

    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        self.options = options
        names = effective_policy_names(options)
        self.policies = {}
        self.learner = None
        for name in sorted(names):
            if name == "ppo":
                from smartsom.learning.base_ppo import BasePPO

                self.learner = BasePPO(
                    options,
                    seed,
                    training=training,
                    selection=selection,
                    numerical_threads=numerical_threads,
                )
                self.policies[name] = self.learner
            elif name in BUILTINS:
                self.policies[name] = BaseRule(name, seed)
            else:
                self.policies[name] = _provider(options.providers[name], seed)
        self.training = bool(training and self.learner is not None)
        self.bound = {
            tuple(point.split(".")): self.policies[
                options.bindings.get(point, options.method)
            ]
            for point in DECISION_POINTS
        }
        self.learning = {
            key
            for key, policy in self.bound.items()
            if self.training and policy is self.learner
        }

    def needs_critic(self, request):
        return (request.group, request.stage) in self.learning

    def choose(self, request, critic=None):
        policy = self.bound[(request.group, request.stage)]
        if policy is self.learner or isinstance(policy, BaseRule):
            return policy.choose(request, critic)
        identity = policy.choose(request)
        for candidate in request.candidates:
            if candidate.identity == identity:
                return candidate.action
        raise ValueError(
            f"provider returned an illegal semantic candidate ID: {identity!r}"
        )

    @property
    def updates(self):
        return self.learner.updates if self.learner is not None else 0

    @property
    def last_losses(self):
        return self.learner.last_losses if self.learner is not None else {}

    @property
    def records(self):
        return self.learner.records if self.learner is not None else []

    @records.setter
    def records(self, value):
        if self.learner is not None:
            self.learner.records = value

    def update(self, rewards, *, terminal, final_critic):
        if self.learner is not None:
            return self.learner.update(
                rewards, terminal=terminal, final_critic=final_critic
            )
        return {}

    def _states(self):
        states = {}
        for name, policy in self.policies.items():
            if isinstance(policy, BaseRule):
                states[name] = policy.rng.getstate()
            elif policy is not self.learner:
                states[name] = policy.get_state()
        # Provider state is deliberately a portable JSON value, never core state.
        return json.loads(json.dumps(states, allow_nan=False))

    def save(self, path):
        if self.learner is None:
            raise ValueError("training checkpoint requires a learning policy")
        self.learner.bound_policy_states = self._states()
        self.learner.save(path)

    def load(self, path):
        from smartsom.experiments.reservation_session import tuples

        if self.learner is None:
            raise ValueError("training checkpoint requires a learning policy")
        self.learner.load(path)
        states = self.learner.bound_policy_states
        expected = set(self.policies) - {"ppo"}
        if set(states) != expected:
            raise ValueError("bound policy checkpoint state mismatch")
        if not self.training:
            return  # Evaluation uses its freshly seeded nonlearning providers.
        for name, state in states.items():
            policy = self.policies[name]
            if isinstance(policy, BaseRule):
                policy.rng.setstate(tuples(state))
            else:
                policy.set_state(state)


def build_policy(
    options, seed=0, *, training=True, selection="sample", numerical_threads=1
):
    if options.bindings:
        return BoundBasePolicy(
            options,
            seed,
            training=training,
            selection=selection,
            numerical_threads=numerical_threads,
        )
    if options.method == "ppo":
        from smartsom.learning.base_ppo import BasePPO

        return BasePPO(
            options,
            seed,
            training=training,
            selection=selection,
            numerical_threads=numerical_threads,
        )
    return BaseRule(options.method, seed)
