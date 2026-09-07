"""Bounded direct execution of pinned, unmodified JobShopLab logistics code.

The namespace bootstrap avoids the umbrella Gym/render imports. This tests the
actual state machine, not the Gym API. Run in a subprocess: imports are local to
this validation process. Full raw sub-states are retained as diagnostic output;
upstream labels them potentially semantically incomplete.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path

COMMIT = "764af47cb5ca3ab7666d08ac8b84385207bfffd9"


def probe(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == COMMIT
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    assert importlib.metadata.version("numpy") == "2.5.3"
    source = root / "jobshoplab"
    for name, directory in (
        ("jobshoplab", source),
        ("jobshoplab.state_machine", source / "state_machine"),
        ("jobshoplab.state_machine.core", source / "state_machine/core"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)

    from jobshoplab.state_machine.core.state_machine import (
        get_possible_transitions,
        is_done,
        step,
    )
    from jobshoplab.state_machine.time_machines import jump_to_event
    from jobshoplab.types import instance_config_types as c
    from jobshoplab.types import state_types as s
    from jobshoplab.types.action_types import Action, ActionFactoryInfo
    from jobshoplab.types.config_types import Config, StateMachineConfig

    def buffer(
        key, capacity=sys.maxsize, role=c.BufferRoleConfig.COMPONENT, parent=None
    ):
        return c.BufferConfig(
            key, c.BufferTypeConfig.FLEX_BUFFER, capacity, (), role, parent=parent
        )

    machines = tuple(
        c.MachineConfig(
            f"m-{i}",
            (),
            {("tl-0", "tl-0"): c.DeterministicTimeConfig(0)},
            buffer(f"b-{3 * i}", parent=f"m-{i}"),
            buffer(f"b-{3 * i + 1}", parent=f"m-{i}"),
            1,
            (),
            buffer(f"b-{3 * i + 2}", 1, parent=f"m-{i}"),
        )
        for i in range(2)
    )
    inlet = buffer("b-12", role=c.BufferRoleConfig.INPUT)
    outlet = buffer("b-13", role=c.BufferRoleConfig.OUTPUT)
    vehicle = c.TransportConfig(
        "t-0", c.TransportTypeConfig.AGV, (), (), buffer("b-9", 1)
    )
    job = c.JobConfig(
        "j-0",
        c.Product("p-0", "P"),
        (
            c.OperationConfig("o-0", "m-0", c.DeterministicTimeConfig(2), "tl-0"),
            c.OperationConfig("o-1", "m-1", c.DeterministicTimeConfig(3), "tl-0"),
        ),
        1,
    )
    nodes = ("b-12", "m-0", "m-1", "b-13")
    overrides = {("m-1", "b-12"): 2, ("b-12", "m-0"): 3, ("m-0", "m-1"): 4}
    times = {
        (a, b): c.DeterministicTimeConfig(0 if a == b else overrides.get((a, b), 1))
        for a in nodes
        for b in nodes
    }
    instance = c.InstanceConfig(
        "Independent 2/3 tick route, empty reposition 2, loaded 3/4/1",
        c.ProblemInstanceConfig(c.ProblemInstanceTypeConfig.JOB_SHOP, (job,)),
        c.LogisticsConfig(sys.maxsize, times),
        machines,
        (inlet, outlet),
        (vehicle,),
    )

    def empty(item):
        return s.BufferState(item.id, s.BufferStateState.EMPTY, ())

    state = s.State(
        (
            s.JobState(
                "j-0",
                tuple(
                    s.OperationState(
                        o.id,
                        s.NoTime(),
                        s.NoTime(),
                        o.machine,
                        s.OperationStateState.IDLE,
                    )
                    for o in job.operations
                ),
                inlet.id,
            ),
        ),
        s.Time(0),
        tuple(
            s.MachineState(
                m.id,
                empty(m.buffer),
                s.NoTime(),
                empty(m.prebuffer),
                empty(m.postbuffer),
                s.MachineStateState.IDLE,
                "tl-0",
                (),
                (),
            )
            for m in machines
        ),
        (
            s.TransportState(
                s.TransportStateState.IDLE,
                vehicle.id,
                s.NoTime(),
                empty(vehicle.buffer),
                s.TransportLocation(0, "m-1"),
                (),
                None,
            ),
        ),
        (
            s.BufferState(inlet.id, s.BufferStateState.NOT_EMPTY, (job.id,)),
            empty(outlet),
        ),
    )
    initial = state.asdict()
    config = Config(state_machine=StateMachineConfig(allow_early_transport=False))
    steps = []
    for _ in range(20):
        choices = sorted(
            get_possible_transitions(state, instance, config),
            key=lambda t: t.component_id,
        )
        action = Action(tuple(choices[:1]), ActionFactoryInfo.Valid, jump_to_event)
        result = step("ERROR", instance, config, state, action)
        assert result.success, result
        steps.append({"action": action.asdict(), "result": result.asdict()})
        state = result.state
        if is_done(state, instance):
            break
    else:
        raise AssertionError("external probe did not finish in 20 steps")
    observations = [
        raw
        for row in steps
        for raw in (*row["result"]["sub_states"], row["result"]["state"])
    ]
    arrivals = {}
    for raw in observations:
        arrivals.setdefault(raw["jobs"][0]["location"], []).append(raw["time"]["time"])
    assert 5 in arrivals["b-0"] and 11 in arrivals["b-3"] and 15 in arrivals["b-13"], (
        arrivals
    )
    assert (
        state.time.time == 14
    )  # Preserve and expose the pinned termination-clock difference.
    operations = [o.asdict() for o in state.jobs[0].operations]
    assert [(o["start_time"]["time"], o["end_time"]["time"]) for o in operations] == [
        (5, 7),
        (11, 14),
    ]
    phases = []
    for row in steps:
        if row["action"]["transitions"][0]["component_id"] != "t-0":
            continue
        raw = row["result"]["sub_states"]
        assert [x["transports"][0]["state"] for x in raw] == [
            "Pickup",
            "WaitingPickup",
            "Transit",
            "Outage",
        ]
        phases.append(
            {
                "empty_start": raw[0]["time"]["time"],
                "pickup": raw[1]["time"]["time"],
                "delivery": raw[3]["time"]["time"],
            }
        )
    assert phases == [
        {"empty_start": 0, "pickup": 2, "delivery": 5},
        {"empty_start": 7, "pickup": 7, "delivery": 11},
        {"empty_start": 14, "pickup": 14, "delivery": 15},
    ]
    hashes = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            path = Path(filename).resolve()
            hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {
        "source_commit": commit,
        "source_url": f"https://github.com/proto-lab-ro/jobshoplab/tree/{commit}",
        "source_file_sha256": hashes,
        "packages": {"numpy": importlib.metadata.version("numpy")},
        "instance": instance.asdict(),
        "initial_state": initial,
        "steps": steps,
        "observed_output_delivery": 15,
        "upstream_final_clock": state.time.time,
        "transport_phases": phases,
        "scope": "Direct state machine only; raw sub-states are diagnostics. Final clock reverts to processing completion 14 despite output delivery at 15. No full Gym or action-equivalence claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = probe(args.source.resolve())
    frozen = (
        Path(__file__).resolve().parents[2] / "data/reference/transport/jobshoplab.json"
    )
    assert json.loads(json.dumps(report)) == json.loads(frozen.read_text()), (
        "external source/stages differ from fixed reference"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "delivery": 15, "upstream_final_clock": 14}
        )
    )


if __name__ == "__main__":
    main()
