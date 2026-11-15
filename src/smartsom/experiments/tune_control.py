"""Translate whole-run cooperative requests to Tune's documented SIGINT path."""

import signal

from ray.tune import Callback

from smartsom.experiments import control


class WholeRunControl(Callback):
    def __init__(self, roots):
        self.roots = tuple(roots)
        self.acknowledged = set()
        self.signalled = False

    def setup(self, **info):
        self.acknowledged.clear()
        self.signalled = False

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        if trial.last_result.get("control_boundary"):
            self.acknowledged.add(trial.trial_id)

    def on_step_end(self, iteration, trials, **info):
        if self.signalled or not any(
            control.pause_requested(root) or control.requested(root, fresh=True)
            for root in self.roots
        ):
            return
        active = {trial.trial_id for trial in trials if trial.status == "RUNNING"}
        if active <= self.acknowledged:
            self.signalled = True
            # Tune handles the first SIGINT by persisting experiment/search
            # state and cleaning up workers. No private controller access.
            signal.raise_signal(signal.SIGINT)
