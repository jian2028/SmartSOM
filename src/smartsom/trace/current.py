"""Current physical snapshots for Studio; recorded viewing never executes policies."""

from smartsom.trace.production import Recorder, canonical


class CurrentRecording:
    def __init__(self, directory, *, source=None):
        self.directory, self.source = directory, source
        self.recorder = None
        self.trip_stream = None

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
        self.trip_stream = (self.recorder.directory / "trips.jsonl").open("w")
        self.recorder.manifest["trip_transit"] = {
            "schema": "smartsom.trip-transit/v1",
            "file": "trips.jsonl",
            "scope": "dispatcher_accepted_to_first_legal_port_arrival",
        }

    def append_trip(self, trip):
        self.trip_stream.write(canonical(trip) + "\n")

    def append(self, result, moves):
        self.recorder.append(
            {
                **result,
                "state": self.simulator.snapshot(),
                "actions": {"movers": moves},
            }
        )

    def __exit__(self, kind, error, tb):
        try:
            if self.trip_stream is not None:
                self.trip_stream.close()
            if self.recorder is not None:
                self.recorder.finish(
                    "failed" if error else self.simulator.status,
                    str(error) if error else None,
                )
        finally:
            if self.trip_stream is not None:
                self.trip_stream.close()
