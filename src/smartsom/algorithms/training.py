"""Bind critic/reward components independently of simulator feature layouts."""

from smartsom.config.components import construct, construct_observation
from smartsom.domain.public_facts import freeze
from smartsom.domain.training import (
    BootstrapInputs,
    LearningContext,
    RewardTimeline,
    RewardTransition,
    public_event,
)
from smartsom.engine.decision_facts import DecisionFacts


class TrainingComponents:
    def __init__(self, options):
        self.options = options
        self.critic = (
            construct_observation(options.learning_policy.critic_observation)
            if options.learning_policy
            else None
        )
        self.reward = None
        self.reward_state = None

    def begin(self, protocol):
        self.reward = construct(
            self.options.reward,
            orders=len(protocol.core.demands),
            quality_penalty=self.options.quality_penalty,
        )
        for method in ("begin_episode", "get_state", "set_state"):
            if not callable(getattr(self.reward, method, None)):
                raise ValueError("reward requires begin_episode/get_state/set_state")
        if self.reward_state is not None:
            self.reward.set_state(self.reward_state)
        self.reward.begin_episode()
        self.timeline = RewardTimeline()
        self.subjects = {}

    def value_input(self, protocol, context):
        with protocol.local_facts(), DecisionFacts(protocol, context) as facts:
            return freeze(self.critic(facts, context))

    def observe_critic(self, protocol, request):
        key = request.group, request.owner
        self.subjects[key] = request.stage
        return self.value_input(
            protocol, LearningContext(request.tick, *key, request.stage)
        )

    def snapshot(self, protocol):
        if not getattr(self.reward, "public_snapshots", True):
            return None
        with protocol.local_facts(), DecisionFacts(protocol, None) as facts:
            return facts.snapshot()

    def transition(self, protocol, start, before, actions, outcome, *, end_reason=None):
        sim = protocol.core
        transition = RewardTransition(
            start,
            sim.tick,
            tuple(actions),
            before,
            self.snapshot(protocol),
            tuple(public_event(e) for e in outcome["events"])
            if getattr(self.reward, "requires_events", True)
            else (),
            freeze(sim.privileged_reward_components()),
            sim.status == "completed",
            sim.status == "truncated",
            end_reason,
        )
        result = self.reward(transition)
        return self.timeline.append_result(result, sim.tick - start)

    def finish(self, protocol, policy):
        self.reward_state = self.reward.get_state()
        final = BootstrapInputs()
        if not getattr(policy, "training", False):
            return final
        shared = None
        for (group, owner), stage in self.subjects.items():
            if shared is None:
                value = self.value_input(
                    protocol,
                    LearningContext(protocol.core.tick, group, owner, stage, True),
                )
                if getattr(self.critic, "subject_independent", False):
                    shared = value
            else:
                value = shared
            final[group, owner] = value
        if (
            self.options.control == "centralized"
            and protocol.core.status != "completed"
            and getattr(getattr(policy, "learner", policy), "needs_next_request", False)
        ):
            from smartsom.engine.central_control import CentralDecision

            request = CentralDecision(
                protocol.core.tick, protocol.central.candidates(), None
            )
            with protocol.local_facts(), DecisionFacts(protocol, request) as facts:
                request = protocol.observation_transform(facts, request)
            self.timeline.final_requests["central", "factory"] = request
        return final
