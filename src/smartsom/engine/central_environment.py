"""Framework-free centralized environment over the shared physical simulator."""

from smartsom.algorithms.base_rules import SharedMover
from smartsom.engine.central_control import ADVANCE, CentralControl
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.public_facts import public_facts


class CentralEnvironment:
    def __init__(self, scenario, options):
        self.core = ProductionSimulator(scenario, contract="base")
        self.core.protocol.configure(options)
        self.control = CentralControl(self.core)
        self.mover = SharedMover(self.core.protocol.routes)

    def observation(self):
        return public_facts(self.core)

    def info(self, physical_dt=0):
        return {
            "candidates": self.control.candidates() if not self.core.done else (),
            "physical_dt": physical_dt,
            "advance_action": ADVANCE.identity,
        }

    def step(self, identity):
        if self.core.done:
            raise ValueError("episode finished; reset before another step")
        selected = self.control.choose(identity)
        reward, dt = 0.0, 0
        if not selected:
            self.control.start_services()
            moves = self.mover.moves(self.core.protocol.traffic_view(), self.core.tick)
            outcome = self.core.protocol.advance(moves)
            if outcome["rejections"]:
                raise AssertionError(outcome["rejections"])
            reward, dt = outcome["reward"], outcome["elapsed"]
            self.core.events = []
            self.control.reconcile()
        return (
            self.observation(),
            reward,
            self.core.status == "completed",
            self.core.status == "truncated",
            self.info(dt),
        )
