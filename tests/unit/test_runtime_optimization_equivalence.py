"""Integrated runtime optimizations preserve real native DQN training behavior.

This is a bounded engineering regression, not a performance or research result.
The reference restores the original snapshot, lookup and list-replay operations;
collection, physical-time targets and optimization stay native.
"""

import copy
import json
import pickle
import random
import struct
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import prepare
from smartsom.config.experiment_v3 import RewardSpecV3, ShipmentTaskReward
from smartsom.engine.port_lookup import PortLookup, decode_target
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.production_protocol import ProductionProtocol
from smartsom.engine.transaction_state import TransactionState
from smartsom.experiments.composable import TrainingSession, allocate

ROOT = Path(__file__).resolve().parents[2]


class _LegacyReplay:
    """Frozen pre-optimization list replay, including sampling and copy behavior."""

    def __init__(self, capacity, seed):
        self.capacity, self.rows, self.position = capacity, [], 0
        self.rng = random.Random(seed)
        self.insertions = 0

    def add(self, row):
        self.insertions += 1
        row = dict(row, diagnostic_insertion=self.insertions)
        if len(self.rows) < self.capacity:
            self.rows.append(copy.deepcopy(row))
        else:
            self.rows[self.position] = copy.deepcopy(row)
        self.position = (self.position + 1) % self.capacity

    def sample(self, count):
        return self.rng.sample(self.rows, count)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "insertions": self.insertions,
            "rows": self.rows,
            "position": self.position,
            "random": self.rng.getstate(),
        }

    def load_state_dict(self, value):
        if self.capacity != value["capacity"]:
            raise ValueError("replay capacity changed during resume")
        self.rows, self.position = copy.deepcopy(value["rows"]), value["position"]
        self.insertions = value.get("insertions", 0)
        self.rng.setstate(value["random"])


def _legacy_capture(self, core):
    return copy.deepcopy({k: v for k, v in vars(core).items() if k != "protocol"})


def _legacy_target(target):
    data = asdict(target)
    owner = next(v for k, v in data.items() if k.endswith("_id") and k != "slot_id")
    return owner, data.get("slot_id", "pool")


def _legacy_ports_for(self, owner, operation):
    return tuple(
        port
        for port in self.ports.values()
        if any(
            self.core._target(binding.target)[0] == owner
            and operation in binding.operations
            for binding in port.bindings
        )
    )


def _prepared(output, task_window):
    config = api.load_config(ROOT / "configs/test/runs/train_all_dqn.yaml")
    config.training.total_ticks = 24
    config.training.ticks_per_update = 8
    config.training.record_initial = False
    if task_window:
        config.training.gamma = 1
        config.training.reward = RewardSpecV3(task=ShipmentTaskReward())
    config.validation.enabled = False
    config.output.root = str(output)
    # Ends an episode between updates and exercises the real truncation probe.
    config.scenario_overrides["tick_limit"] = 10
    config.runtime.num_envs = 1
    config.runtime.sampling_processes = 0
    config.runtime.numerical_threads = 1
    config.runtime.device = "cpu"
    prepared = prepare(config)
    parameters = json.loads(prepared.parameters_json)
    parameters.update(
        batch_size=2,
        replay_capacity=16,
        warmup_ticks=0,
        train_every_ticks=1,
        gradient_steps=1,
        target_update_ticks=5,
    )
    return replace(prepared, parameters_json=canonical_json(parameters))


def _place_active_inventory(session):
    """Valid hand boundary with Machine work and a two-job Buffer prefix."""
    sim = session.sims[0]
    job = next(iter(sim.jobs))
    operation = sim.demands[sim.jobs[job]["demand"]].steps[0].operation_type
    machine = next(
        owner
        for owner, design in sim.machines.items()
        if operation in design.operation_types
    )
    jobs = [
        job
        for job, state in sim.jobs.items()
        if sim.demands[state["demand"]].steps[0].operation_type == operation
    ]
    assert len(jobs) >= 3
    for job in jobs[:3]:
        slot = sim._free(sim.pre[machine])
        assert slot is not None
        sim._remove(job)
        sim._place(job, sim.pre[machine], slot)

    source = next(owner for owner, role in sim.roles.items() if role == "system_input")
    ports = sim.protocol.ports_for(source, "pickup")
    assert len(ports) >= 2 and len(sim.protocol.ready(source)) >= 3
    cars = list(sim.agvs)[:2]
    for car, port in zip(cars, ports[:2], strict=True):
        sim.agvs[car].update(
            cell=[port.cell.x, port.cell.y],
            target={"owner": source, "port": port.port_id},
            reservation=source,
            reservation_tick=0,
            arrived_at=0,
        )
        if sim.protocol.matrix:
            sim.agvs[car].update(point=port.port_id, travel=None)
        assert sim.protocol.arrived(sim.agvs[car], port)
    sim._check()


def _behavior_state(session, batches):
    state = session.state_dict()
    # Framework metrics include wall-clock timings. Keep its actual module,
    # target, optimizer and counters plus all native learner diagnostics.
    state["learners"] = {
        group: {key: value for key, value in saved.items() if key != "metrics_logger"}
        for group, saved in state["learners"].items()
    }
    # Compact checkpoint bytes intentionally differ. Compare every decoded row,
    # ring cursor and replay RNG, as well as exact bytes passed to the learner.
    state["replays"] = {
        group: {
            "capacity": replay.capacity,
            "insertions": replay.insertions,
            "rows": list(replay.rows),
            "position": replay.position,
            "random": replay.rng.getstate(),
        }
        for group, replay in session.replays.items()
    }
    state["learner_batch_bytes"] = batches
    # Compare the complete private simulator graph and protocol scratch values.
    # Only the implementation-specific acceleration plans differ by design.
    state["sims"] = [
        {
            "core": {k: v for k, v in vars(sim).items() if k != "protocol"},
            "protocol": {
                k: v
                for k, v in vars(sim.protocol).items()
                if k not in {"core", "state_plan", "_port_lookup"}
            },
        }
        for sim in session.sims
    ]
    return copy.deepcopy(state)


def _assert_equal(actual, expected, path="state"):
    import numpy as np
    import torch

    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        ), path
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        assert actual.tobytes() == expected.tobytes(), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal(a, b, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert struct.pack("!d", actual) == struct.pack("!d", expected), path
    else:
        assert actual == expected, path


def _array_bytes(value):
    if isinstance(value, dict):
        return {key: _array_bytes(row) for key, row in value.items()}
    return value.dtype.str, value.shape, value.tobytes()


@pytest.mark.parametrize(
    ("task_window", "active_inventory"),
    [(False, False), (True, False), (True, True)],
    ids=["bootstrap", "terminal", "active_inventory"],
)
def test_native_dqn_runtime_matches_legacy_and_restores_exactly(
    tmp_path, monkeypatch, task_window, active_inventory
):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("ray.rllib")
    from smartsom.learning import production_collection, production_rllib_v3
    from smartsom.learning.production_rllib_v3 import PhysicalDQNLearner

    original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    original_cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    original_threads = torch.get_num_threads()
    native_capture = TransactionState.capture
    prepared = _prepared(tmp_path / "runs", task_window)
    native_replay_batch = production_rllib_v3.replay_batch
    checkpoints, runs = {}, {}
    # The active-inventory checkpoint contains trained Machine/Buffer Adam state
    # and nonempty Buffer prefixes, rather than only their pending first choices.
    checkpoint_update = 2 if active_inventory else 1

    def record_batches(patch, batches):
        def replay_batch(*args, **kwargs):
            batch = native_replay_batch(*args, **kwargs)
            batches.append(_array_bytes(dict(batch.policy_batches["default_policy"])))
            return batch

        patch.setattr(production_rllib_v3, "replay_batch", replay_batch)

    def run(legacy):
        label = "legacy" if legacy else "integrated"
        calls = Counter()

        def capture(plan, core):
            calls["capture"] += 1
            if legacy:
                return _legacy_capture(plan, core)
            saved = native_capture(plan, core)
            calls["fast_capture"] += int(saved.fast_path)
            return saved

        def target(value):
            calls["asdict"] += 1
            return _legacy_target(value)

        def ports_for(protocol, owner, operation):
            calls["scan"] += 1
            return _legacy_ports_for(protocol, owner, operation)

        with monkeypatch.context() as patch:
            batches = []
            record_batches(patch, batches)
            patch.setattr(TransactionState, "capture", capture)
            if legacy:
                patch.setattr(production_collection, "Replay", _LegacyReplay)
                patch.setattr(ProductionSimulator, "_target", staticmethod(target))
                patch.setattr(ProductionProtocol, "ports_for", ports_for)
            else:
                assert ProductionSimulator._target is decode_target
            root, record, frozen = allocate(prepared, "training")
            random.seed(701)
            np.random.seed(702)
            torch.manual_seed(703)
            session = TrainingSession(frozen, root, record)
            try:
                if active_inventory:
                    _place_active_inventory(session)
                    prefix_before = {
                        key: value.detach().clone()
                        for key, value in session.policies["buffer"]
                        .network.actor_prefix.state_dict()
                        .items()
                    }
                assert all(
                    isinstance(learner, PhysicalDQNLearner)
                    for learner in session.learners.values()
                )
                states = [_behavior_state(session, batches)]
                for update in range(1, 4):
                    session.step_update()
                    assert session.updates == update and session.ticks == update * 8
                    for sim in session.sims:
                        assert sim.shipped is sim.completed
                        assert sim.protocol.stage == "closed"
                        if not legacy:
                            assert isinstance(sim.protocol._port_lookup, PortLookup)
                    states.append(_behavior_state(session, batches))
                    if update == checkpoint_update:
                        checkpoint = (
                            root / f"checkpoints/update-{update:06d}/continuation.pkl"
                        )
                        checkpoints[label] = (
                            frozen,
                            root,
                            record,
                            checkpoint.read_bytes(),
                        )
                assert session.training_done
                assert len(session.actions) == 24
                if not task_window:
                    assert any(row["reward"] != 0 for row in session.actions)
                assert session.episodes == [2]
                active = [g for g, count in session.optimizations.items() if count]
                assert active and batches, "must execute actual native optimization"
                assert any(
                    session.policies[g].fingerprint() != session.initial[g]
                    for g in active
                )
                assert all(session.target_clock[g] == 20 for g in session.learners)
                assert any(r.insertions > r.capacity for r in session.replays.values())
                for group in active:
                    assert states[-1]["learners"][group]["optimizer"]
                if active_inventory:
                    assert {"machine", "buffer"} <= set(active)
                    for group in ("machine", "buffer"):
                        assert (
                            session.policies[group].fingerprint()
                            != session.initial[group]
                        )
                        assert states[checkpoint_update]["optimizations"][group] > 0
                    buffer_rows = list(session.replays["buffer"].rows)
                    assert any(row["input"]["prefix"] for row in buffer_rows)
                    assert any(row["dt"] == 0 for row in buffer_rows)
                    assert any(row["dt"] > 0 for row in buffer_rows)
                    assert any(
                        np.frombuffer(
                            batch["obs"]["prefix_length"][2],
                            dtype=batch["obs"]["prefix_length"][0],
                        ).max()
                        > 0
                        for batch in batches
                    ), "a nonempty Buffer prefix must reach the actual learner"
                    assert any(
                        not torch.equal(value, prefix_before[key])
                        for key, value in session.policies["buffer"]
                        .network.actor_prefix.state_dict()
                        .items()
                    ), "the native Buffer GRU must receive a real parameter update"
                assert (
                    any(
                        row["terminated"]
                        for replay in session.replays.values()
                        for row in replay.rows
                    )
                    is task_window
                )
                if not task_window:
                    assert calls["capture"] > 24  # Stage-only truncation rollback.
                if legacy:
                    assert calls["asdict"] and calls["scan"]
                else:
                    assert calls["fast_capture"] == calls["capture"]
                return states
            finally:
                session.close()

    try:
        runs["legacy"] = run(True)
        runs["integrated"] = run(False)
        for update, (actual, expected) in enumerate(
            zip(runs["integrated"], runs["legacy"], strict=True)
        ):
            _assert_equal(actual, expected, f"update[{update}]")

        # Both a legacy-operation checkpoint and an optimized checkpoint must
        # continue under the integrated runtime with exact learner/replay/RNG state.
        for label, (frozen, root, record, checkpoint) in checkpoints.items():
            with monkeypatch.context() as patch:
                batches = copy.deepcopy(
                    runs["integrated"][checkpoint_update]["learner_batch_bytes"]
                )
                record_batches(patch, batches)
                resumed = TrainingSession(frozen, root, copy.deepcopy(record))
                try:
                    resumed.restore(pickle.loads(checkpoint))
                    _assert_equal(
                        _behavior_state(resumed, batches),
                        runs["integrated"][checkpoint_update],
                        f"{label}.restore",
                    )
                    for update in range(checkpoint_update + 1, 4):
                        resumed.step_update()
                        _assert_equal(
                            _behavior_state(resumed, batches),
                            runs["integrated"][update],
                            f"{label}.continued[{update}]",
                        )
                finally:
                    resumed.close()
    finally:
        random.setstate(original_rng[0])
        np.random.set_state(original_rng[1])
        torch.set_rng_state(original_rng[2])
        if original_cuda_rng is not None:
            torch.cuda.set_rng_state_all(original_cuda_rng)
        torch.set_num_threads(original_threads)
