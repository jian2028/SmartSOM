"""Framework-free centralized environment over the shared physical simulator."""

from smartsom.engine.central_control import ADVANCE, CentralControl
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.public_facts import public_facts
from smartsom.traffic import build_mover


class CentralEnvironment:
    def __init__(self, scenario, options):
        self.core = ProductionSimulator(scenario, contract="base")
        self.core.protocol.configure(options)
        self.control = CentralControl(self.core, options.central)
        self.mover = build_mover(options, self.core.protocol.routes)

    def observation(self):
        return public_facts(self.core)

    def info(self, physical_dt=0):
        candidates = self.control.candidates() if not self.core.done else ()
        return {
            "candidates": candidates,
            "physical_dt": physical_dt,
            # None while advance_only_when_idle hides ADVANCE behind work.
            "advance_action": ADVANCE.identity
            if not candidates or ADVANCE in candidates
            else None,
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
