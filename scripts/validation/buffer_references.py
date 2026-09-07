"""Independent, bounded comparisons; neither reference changes SmartSOM's state.

PyJobShop checks a zero-buffer resource-holding interval. Pinned JobShopLab
checks capacity/flex selection and records its full/zero postbuffer errors.
The latter does not implement our blocking or AGV reservation semantics.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

COMMIT = "764af47cb5ca3ab7666d08ac8b84385207bfffd9"


def pyjobshop_reference():
    from pyjobshop import Model

    for name, version in [("pyjobshop", "0.0.9"), ("ortools", "9.12.4544")]:
        assert importlib.metadata.version(name) == version
    model = Model()
    machines = {key: model.add_machine(name=key) for key in ("M1", "M2")}
    specs = [
        ("A1", "M1", 2, 0),
        ("A2", "M2", 1, 5),
        ("B1", "M1", 1, 5),
        ("C1", "M2", 5, 0),
    ]
    tasks = {}
    for key, machine, duration, start in specs:
        task = model.add_task(
            name=key, earliest_start=start, latest_start=start, allow_idle=key == "A1"
        )
        model.add_mode(task, machines[machine], duration)
        tasks[key] = task
    model.add_end_at_start(tasks["A1"], tasks["A2"])
    solved = model.solve(
        solver="ortools", time_limit=10, num_workers=1, random_seed=0, display=False
    )
    rows = {
        spec[0]: {
            "start": t.start,
            "resource_release": t.end,
            "processing": t.processing,
            "idle": t.idle,
        }
        for spec, t in zip(specs, solved.best.tasks)
    }
    assert solved.status.name == "OPTIMAL"
    assert solved.objective == solved.lower_bound == 6
    assert rows == {
        "A1": {"start": 0, "resource_release": 5, "processing": 2, "idle": 3},
        "A2": {"start": 5, "resource_release": 6, "processing": 1, "idle": 0},
        "B1": {"start": 5, "resource_release": 6, "processing": 1, "idle": 0},
        "C1": {"start": 0, "resource_release": 5, "processing": 5, "idle": 0},
    }
    import pyjobshop

    source = Path(pyjobshop.__file__).parent
    return {
        "source": "https://pyjobshop.org/stable/examples/permutation_flow_shop.html#blocking",
        "packages": {
            n: importlib.metadata.version(n) for n in ("pyjobshop", "ortools")
        },
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))
        },
        "status": solved.status.name,
        "objective": solved.objective,
        "bound": solved.lower_bound,
        "tasks": rows,
        "scope": "Matched fixed-start micro case using allow_idle and end_at_start. Task end is resource release, not net processing completion; no transport, reservation, or dynamic equivalence claim.",
    }


def jobshoplab_reference(root):
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        == COMMIT
    )
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    assert importlib.metadata.version("numpy") == "2.5.3"
    source = root / "jobshoplab"
    for name, directory in [
        ("jobshoplab", source),
        ("jobshoplab.state_machine", source / "state_machine"),
        ("jobshoplab.state_machine.core", source / "state_machine/core"),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)
    from jobshoplab.state_machine.core.state_machine.manipulate import (
        complete_active_operation_on_machine,
    )
    from jobshoplab.types import instance_config_types as c
    from jobshoplab.types import state_types as s
    from jobshoplab.utils.exceptions import BufferFullError
    from jobshoplab.utils.state_machine_utils.buffer_type_utils import (
        put_in_buffer,
        remove_from_buffer,
    )

    def buf(key, capacity):
        return c.BufferConfig(
            key,
            c.BufferTypeConfig.FLEX_BUFFER,
            capacity,
            (),
            c.BufferRoleConfig.COMPONENT,
            parent="m-0",
        )

    job = s.JobState(
        "j-0",
        (
            s.OperationState(
                "o-0", s.Time(0), s.Time(2), "m-0", s.OperationStateState.PROCESSING
            ),
        ),
        "b-2",
    )
    pre, post, slot = buf("b-0", 2), buf("b-1", 1), buf("b-2", 1)

    def empty(b):
        return s.BufferState(b.id, s.BufferStateState.EMPTY, ())

    filled, _ = put_in_buffer(empty(pre), pre, job)
    filled, _ = put_in_buffer(filled, pre, replace(job, id="j-1"))
    # FLEX pickup of the second entrant leaves the first untouched.
    after = remove_from_buffer(filled, "j-1")
    assert filled.store == ("j-0", "j-1") and after.store == ("j-0",)
    machine = c.MachineConfig(
        "m-0",
        (),
        {("tl-0", "tl-0"): c.DeterministicTimeConfig(0)},
        pre,
        post,
        1,
        (),
        slot,
    )
    jc = c.JobConfig(
        "j-0",
        c.Product("p-0", "P"),
        (c.OperationConfig("o-0", "m-0", c.DeterministicTimeConfig(2), "tl-0"),),
        1,
    )
    instance = c.InstanceConfig(
        "Independent capacity/blocking boundary",
        c.ProblemInstanceConfig(c.ProblemInstanceTypeConfig.JOB_SHOP, (jc,)),
        c.LogisticsConfig(sys.maxsize, {}),
        (machine,),
        (),
        (),
    )
    ms = s.MachineState(
        "m-0",
        s.BufferState(slot.id, s.BufferStateState.FULL, ("j-0",)),
        s.Time(2),
        empty(pre),
        empty(post),
        s.MachineStateState.WORKING,
        "tl-0",
        (),
        (),
    )
    cases = {}
    for name, capacity, store in [
        ("space", 1, ()),
        ("full", 1, ("other",)),
        ("zero", 0, ()),
    ]:
        ci = replace(
            instance,
            machines=(replace(machine, postbuffer=replace(post, capacity=capacity)),),
        )
        state = replace(ms, postbuffer=replace(ms.postbuffer, store=store))
        before = state.asdict()
        try:
            j, m = complete_active_operation_on_machine(ci, (job,), state, s.Time(2))
            assert (
                name == "space"
                and not m.buffer.store
                and m.postbuffer.store == ("j-0",)
            )
            cases[name] = {"job": j.asdict(), "machine": m.asdict()}
        except BufferFullError as exc:
            assert name in ("full", "zero")
            cases[name] = {
                "exception": type(exc).__name__,
                "message": str(exc),
                "input_state": before,
            }
        assert state.asdict() == before
    hashes = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            path = Path(filename).resolve()
            hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {
        "source_commit": COMMIT,
        "source": f"https://github.com/proto-lab-ro/jobshoplab/tree/{COMMIT}",
        "source_sha256": hashes,
        "packages": {"numpy": importlib.metadata.version("numpy")},
        "instance": instance.asdict(),
        "initial_job": job.asdict(),
        "initial_machine": ms.asdict(),
        "flex": {"before": filled.asdict(), "after_second_pickup": after.asdict()},
        "completion": cases,
        "scope": "Actual capacity and flex-buffer helpers plus operation completion. Full and zero postbuffers raise BufferFullError, rather than retaining a completed-blocked job. No complete blocking or reservation equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "pyjobshop": pyjobshop_reference(),
        "jobshoplab": jobshoplab_reference(args.source.resolve()),
    }
    frozen = (
        Path(__file__).resolve().parents[2] / "data/reference/buffers/external.json"
    )
    if frozen.exists():
        assert json.loads(json.dumps(report)) == json.loads(frozen.read_text()), (
            "external evidence differs from frozen reference"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "objective": report["pyjobshop"]["objective"],
                "matched_cases": ["capacity", "flex", "zero-buffer resource holding"],
            }
        )
    )


if __name__ == "__main__":
    main()
