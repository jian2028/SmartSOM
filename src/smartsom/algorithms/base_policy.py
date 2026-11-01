"""Startup-bound semantic decisions over explicit policy and learner components."""

import json

from smartsom.config.components import construct, instantiate, resolve

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
BUILTINS = frozenset(
    {"rule", "random", "ppo", "dqn"}
)  # migration/evaluation recipes only


def effective_policy_names(options):
    points = {"central.assign"} if options.control == "centralized" else DECISION_POINTS
    return {
        options.production.bindings.get(point, options.production.default)
        for point in points
    }


def uses_learning(options):
    return any(
        options.production.policies[n].kind == "learning"
        for n in effective_policy_names(options)
    )


def require_methods(instance, methods, label):
    missing = [name for name in methods if not callable(getattr(instance, name, None))]
    if missing:
        raise ValueError(f"{label} requires {', '.join(missing)}")


class BoundBasePolicy:
    """One learning instance plus explicit nonlearning providers and bindings."""

    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        self.options = options
        self.policies = {}
        self.learner = None
        self.learner_name = None
        for name in sorted(effective_policy_names(options)):
            spec = options.production.policies[name]
            if spec.kind == "learning":
                policy = construct(
                    spec.learner,
                    options=options,
                    seed=seed,
                    training=training,
                    selection=selection,
                    numerical_threads=numerical_threads,
                )
                require_methods(
                    policy,
                    (
                        "choose",
                        "needs_critic",
                        "update",
                        "get_state",
                        "set_state",
                        "sampling_state",
                        "load_sampling_state",
                        "seed_sampling",
                        "discard_rollout",
                    ),
                    "learner",
                )
                self.learner, self.learner_name = policy, name
            else:
                policy = construct(
                    spec.implementation, seed=seed, control=options.control
                )
                require_methods(policy, ("choose", "get_state", "set_state"), "policy")
            self.policies[name] = policy
        from smartsom.algorithms.observation import build_observation

        self.observation_transform = build_observation(options)
        self._mover_factory = resolve(options.traffic.mover)
        self.training = bool(training and self.learner is not None)
        points = (
            {"central.assign"} if options.control == "centralized" else DECISION_POINTS
        )
        self.bound = {
            tuple(p.split(".")): self.policies[
                options.production.bindings.get(p, options.production.default)
            ]
            for p in points
        }

    def create_mover(self, routes, seed):
        from smartsom.traffic import validate_mover

        mover = instantiate(
            self._mover_factory,
            self.options.traffic.mover.params,
            routes=routes,
            seed=seed,
        )
        return validate_mover(mover)

    def needs_critic(self, request):
        policy = self.bound[
            ("central", "assign")
            if self.options.control == "centralized"
            else (request.group, request.stage)
        ]
        return bool(policy is self.learner and policy.needs_critic(request))

    def choose(self, request, critic=None):
        policy = self.bound[
            ("central", "assign")
            if self.options.control == "centralized"
            else (request.group, request.stage)
        ]
        chosen = policy.choose(request, critic)
        for candidate in request.candidates:
            if self.options.control == "centralized":
                if chosen == candidate.identity:
                    return candidate.identity
            elif chosen == candidate.identity or chosen == candidate.action:
                return candidate.action
        raise ValueError(f"policy returned an illegal semantic action: {chosen!r}")

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

    def discard_rollout(self):
        if self.learner is not None:
            self.learner.discard_rollout()

    def seed_sampling(self, seed):
        if self.learner is not None:
            self.learner.seed_sampling(seed)

    def update(self, rewards, *, terminal, final_critic=None):
        return (
            self.learner.update(rewards, terminal=terminal, final_critic=final_critic)
            if self.learner is not None
            else {}
        )

    def get_state(self):
        if self.learner is None:
            raise ValueError("training checkpoint requires a learning policy")
        return {
            "learner": self.learner.get_state(),
            "policies": json.loads(
                json.dumps(
                    {
                        n: p.get_state()
                        for n, p in self.policies.items()
                        if n != self.learner_name
                    },
                    allow_nan=False,
                )
            ),
        }

    def set_state(self, state):
        expected = set(self.policies) - {self.learner_name}
        if set(state["policies"]) != expected:
            raise ValueError("bound policy checkpoint state mismatch")
        self.learner.set_state(state["learner"])
        if self.training:
            for name in expected:
                self.policies[name].set_state(state["policies"][name])

    def save(self, path):
        from smartsom.learning.state import save

        save(path, self.options, self.get_state())

    def load(self, path):
        from smartsom.learning.state import load

        self.set_state(load(path, self.options))


def build_policy(
    options, seed=0, *, training=True, selection="sample", numerical_threads=1
):
    return BoundBasePolicy(
        options,
        seed,
        training=training,
        selection=selection,
        numerical_threads=numerical_threads,
    )


def evaluation_recipe(options, method):
    """Named comparison recipes preserve the configured protocol and traffic."""
    if method == "configured":
        return options
    from smartsom.config.base_diagnostic import BaseAlgorithm, ProductionPolicies

    if method in {"rule", "random"}:
        production = {
            "default": method,
            "policies": {
                method: {
                    "kind": "policy",
                    "implementation": {
                        "target": f"smartsom.policies.{method}.create_policy"
                    },
                }
            },
        }
        return BaseAlgorithm.model_validate(
            {**options.model_dump(mode="json", by_alias=True), "production": production}
        )
    if options.learning_policy is not None:
        target = options.learning_policy.learner.target
        if target != f"smartsom.learning.{method}.create_learner":
            raise ValueError(
                "evaluation recipe differs from configured learner; use configured"
            )
        return options
    if options.control == "centralized":
        raise ValueError("centralized learning is unsupported")
    production = ProductionPolicies.model_validate(
        {
            "default": method,
            "policies": {
                method: {
                    "kind": "learning",
                    "learner": {"target": f"smartsom.learning.{method}.create_learner"},
                    "observation": {
                        "target": "smartsom.representations.base.observation.create_observation"
                    },
                    "encoder": {
                        "target": "smartsom.representations.base.encoder.create_encoder"
                    },
                }
            },
        }
    )
    return options.model_copy(update={"production": production})
