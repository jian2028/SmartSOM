"""Semantic factory stepping for the RLlib adapters, without framework imports.

Production requests retain the original generator order. A learned WAIT frame
separates exhausted negotiation from the existing physical advance; it never
inserts an additional physical interval.
"""

from dataclasses import dataclass

from smartsom.algorithms.observation import build_observation
from smartsom.algorithms.training import TrainingComponents
from smartsom.domain.training import CommittedAction, LearningContext
from smartsom.engine.production import ProductionSimulator
from smartsom.rewards.normalized_task import cost_snapshot, task_metrics
from smartsom.traffic import build_mover


@dataclass(frozen=True, slots=True)
class FactoryFrame:
    tick: int
    request: object | None
    waiting: tuple[LearningContext, ...]
    team_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


class FactoryEpisode:
    """One physical realization; actions are semantic candidate identities."""

    def __init__(self, scenario, options, *, action_seed=0):
        self.options = options
        self.sim = ProductionSimulator(scenario, contract="base")
        self.protocol = self.sim.protocol
        self.protocol.configure(options)
        if options.control == "centralized":
            from smartsom.engine.central_control import CentralControl

            CentralControl(self.sim, options.central)
        self.protocol.observation_transform = build_observation(options)
        self.mover = build_mover(options, self.protocol.routes, action_seed)
        self.components = TrainingComponents(options)
        self.components.begin(self.protocol)
        self.normalized_reward = (
            options.control == "base"
            and options.reward.target == "smartsom.rewards.default.create_reward"
        )
        self.subjects = {}
        self.actions = []
        self.decisions = 0
        self.wait_steps = 0
        self.team_return = 0.0
        self._generator = self.protocol.decisions()
        self.frame = self._next()

    def subject_key(self, request):
        point = request.group + "." + request.stage
        name = self.options.production.bindings.get(
            point, self.options.production.default
        )
        policy = self.options.production.policies[name]
        stage = request.stage if request.group in policy.split_stages else ""
        return request.group, request.owner, stage

    def _waiting(self):
        return tuple(
            LearningContext(self.sim.tick, row.group, row.owner, row.stage)
            for row in self.subjects.values()
        )

    def _next(self, action=None, *, send=False, reward=0.0):
        try:
            request = self._generator.send(action) if send else next(self._generator)
        except StopIteration:
            self._generator = None
            return FactoryFrame(self.sim.tick, None, self._waiting(), reward)
        self.subjects[self.subject_key(request)] = request
        return FactoryFrame(self.sim.tick, request, (), reward)

    def act(self, identity):
        request = self.frame.request
        if request is None or self.frame.terminated or self.frame.truncated:
            raise ValueError("no pending production decision")
        selected = next((c for c in request.candidates if c.identity == identity), None)
        if selected is None:
            raise ValueError("action is not a current legal semantic candidate")
        self.actions.append(
            CommittedAction(
                request.tick,
                request.group,
                request.owner,
                request.stage,
                selected.identity,
                selected.action,
            )
        )
        self.decisions += 1
        self.frame = self._next(selected.action, send=True)
        return self.frame

    def wait(self):
        if (
            self.frame.request is not None
            or self.frame.terminated
            or self.frame.truncated
        ):
            raise ValueError("WAIT is legal only at a pending physical boundary")
        before = cost_snapshot(self.sim)
        reward_before = self.components.snapshot(self.protocol)
        moves = self.mover.moves(self.protocol.traffic_view(), self.sim.tick)
        outcome = self.protocol.advance(moves)
        if outcome["rejections"]:
            raise AssertionError(
                f"shared Mover violated legality: {outcome['rejections']}"
            )
        if self.normalized_reward:
            reward = cost_snapshot(self.sim).reward_since(before)
        else:
            reward = self.components.transition(
                self.protocol, before.elapsed, reward_before, self.actions, outcome
            )["team"]
        self.actions.clear()
        self.wait_steps += 1
        self.team_return += reward
        complete = len(self.sim.completed) == len(self.sim.demands)
        if complete or self.sim.done:
            self.frame = FactoryFrame(
                self.sim.tick,
                None,
                self._waiting(),
                reward,
                terminated=complete,
                truncated=not complete,
            )
        else:
            self._generator = self.protocol.decisions()
            self.frame = self._next(reward=reward)
        return self.frame

    def metrics(self):
        return {
            **task_metrics(self.sim),
            "learning_return": self.team_return,
            "production_decisions": self.decisions,
            "physical_wait_steps": self.wait_steps,
        }

    def close(self):
        if self._generator is not None:
            self._generator.close()
            self._generator = None
