"""Current physical snapshots for Studio; recorded viewing never executes policies."""

from smartsom.trace.production import Recorder


class CurrentRecording:
    def __init__(self, directory, *, source=None):
        self.directory, self.source = directory, source
        self.recorder = None

    def __enter__(self):
        return self

    def begin(self, simulator):
        self.simulator = simulator
        self.recorder = Recorder(
            self.directory,
            {"scenario": simulator.scenario, "algorithm": {"provider": "current"}},
            simulator.snapshot(),
            source=self.source,
        )
        self.recorder.manifest["action_contract"] = "smartsom.current-actions/v1"

    def append(self, result, moves):
        self.recorder.append(
            {
                **result,
                "state": self.simulator.snapshot(),
                "actions": {"movers": moves},
            }
        )

    def __exit__(self, kind, error, tb):
        if self.recorder is not None:
            self.recorder.finish(
                "failed" if error else self.simulator.status,
                str(error) if error else None,
            )
