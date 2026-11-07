"""Current physical snapshots for Studio; recorded viewing never executes policies."""

from dataclasses import asdict

from smartsom.trace.production import Recorder, canonical


class CurrentRecording:
    def __init__(self, directory, *, source=None):
        self.directory, self.source = directory, source
        self.recorder = None
        self.trip_stream = None
        self.reward_stream = None
        self.pending_decisions = []

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
        self.reward_stream = (self.recorder.directory / "rewards.jsonl").open("w")
        self.recorder.manifest["learning_rewards"] = {
            "schema": "smartsom.reward-interval/v1",
            "file": "rewards.jsonl",
        }
        self.recorder.manifest["trip_transit"] = {
            "schema": "smartsom.trip-transit/v1",
            "file": "trips.jsonl",
            "scope": "dispatcher_accepted_to_first_legal_port_arrival",
        }

    def reward(self, start, end, values):
        self.reward_stream.write(
            canonical(
                {
                    "start": start,
                    "end": end,
                    "team": values.get("team", 0.0),
                    "roles": values.get("roles", {}),
                    "agents": [
                        {"role": g, "owner": o, "value": v}
                        for (g, o), v in values.get("agents", {}).items()
                    ],
                }
            )
            + "\n"
        )

    def append_trip(self, trip):
        self.trip_stream.write(canonical(trip) + "\n")

    def decision(self, request, action, *, selected_identity=None):
        self.pending_decisions.append(
            {
                "decision": {
                    "tick": request.tick,
                    "group": request.group,
                    "owner": request.owner,
                    "stage": request.stage,
                    "candidates": [
                        {
                            **asdict(c),
                            "action": getattr(c, "action", c.identity),
                        }
                        for c in request.candidates
                    ],
                },
                "action": action,
                **(
                    {"selected_identity": selected_identity}
                    if selected_identity is not None
                    else {}
                ),
            }
        )

    def append(self, result, moves):
        self.recorder.append(
            {
                **result,
                "state": self.simulator.snapshot(),
                "actions": {"movers": moves},
                "decisions": self.pending_decisions,
            }
        )

        self.pending_decisions = []

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
            if self.reward_stream is not None:
                self.reward_stream.close()
            if self.trip_stream is not None:
                self.trip_stream.close()
