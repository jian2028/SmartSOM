import copy
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import torch

from smartsom import api
from smartsom.config.codec import canonical_json, digest
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.composable import (
    _sampling_threads,
    _worker_tick,
    bindings,
    policies_for,
)
from smartsom.learning.physical_job_observation import (
    PhysicalJobEncoder,
    PhysicalJobNetwork,
    _normal_reference,
    install,
)
from smartsom.learning.production_inference import ModelPolicy
from smartsom.learning.production_models import default_network


def main():
    root = Path(os.environ["SMARTSOM_TEST_ROOT"])
    torch.set_num_threads(1)
    identity = install(include_inspection=True)
    p = api.prepare(
        api.load_config(root / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    scene = p.scenario
    buffers = tuple(
        replace(b, storage=replace(b.storage, capacity=10))
        if b.role == "system_input"
        else b
        for b in scene.factory.buffers
    )
    scene = replace(
        scene,
        factory=replace(scene.factory, buffers=buffers),
        demands=tuple(
            replace(
                d,
                steps=tuple(replace(step, reference_ticks="21/2") for step in d.steps),
            )
            for d in scene.demands
        ),
    )
    p = replace(p, scenario_json=canonical_json(scene))
    with ProcessPoolExecutor(
        2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_sampling_threads,
        initargs=(1, (), identity),
    ) as pool:
        for algorithm in ("ppo", "dqn"):
            policies, _ = policies_for(p, training=False)
            encoder = PhysicalJobEncoder(
                scene.factory,
                {"time_scale": 100, "count_scale": 100},
                provider="rllib.resource_" + algorithm,
                role="dispatcher",
            )
            network = PhysicalJobNetwork(
                encoder.context_size,
                default_network(algorithm),
                "rllib.resource_" + algorithm,
                algorithm=algorithm,
                candidate_width=encoder.candidate_width,
            )
            policies["dispatcher"] = ModelPolicy(
                encoder, network, {"role": "dispatcher"}, 101
            )
            payload = pickle.dumps(policies, protocol=5)
            sim = ProductionSimulator(scene, contract="v3")
            serial = _worker_tick(
                copy.deepcopy(sim), payload, bindings(p), "first_arrival", 0
            )
            futures = [
                pool.submit(
                    _worker_tick,
                    copy.deepcopy(sim),
                    payload,
                    bindings(p),
                    "first_arrival",
                    0,
                )
                for _ in range(2)
            ]
            for future in futures:
                actual = future.result(timeout=90)
                assert digest(actual[0].snapshot()) == digest(serial[0].snapshot())
                view = actual[0].protocol.public_view()
                assert view["jobs"] and all(
                    "physical_job_original_reference_work" in row
                    for row in view["jobs"].values()
                    if row["location"] in encoder.physical
                )
                for row in view["jobs"].values():
                    if row["location"] in encoder.physical:
                        demand = next(
                            d for d in scene.demands if d.demand_id == row["demand"]
                        )
                        reference = sum(
                            _normal_reference(asdict(step), scene.factory)
                            for step in demand.steps
                        )
                        assert row["physical_job_original_reference_work"] == reference
                saved = pickle.loads(actual[3])
                assert identity in repr(saved)
    print("REAL_SPAWN_PPO_DQN_SINGLE_AND_TWO_WAVES_PASSED")


def restore_probe():
    import tempfile

    import numpy as np

    from smartsom.experiments.composable import TrainingSession

    torch.set_num_threads(1)
    identity = install(include_inspection=True)
    root = Path(os.environ["SMARTSOM_TEST_ROOT"])

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    for algorithm, environments, workers in (
        (a, e, w) for a in ("ppo", "dqn") for e, w in ((1, 0), (1, 1), (2, 2))
    ):
        config = api.load_config(
            root / ("configs/test/runs/train_all_" + algorithm + ".yaml")
        )
        config.scenario = str(root / "configs/test/scenarios/small_matrix_auto.yaml")
        config.composition = str(
            root / ("configs/test/compositions/small_train_" + algorithm + ".yaml")
        )
        config.training.groups = ("machine", "buffer", "dispatcher")
        config.training.total_ticks = 16
        config.training.ticks_per_update = 8
        config.runtime.num_envs = environments
        config.runtime.sampling_processes = workers
        config.training.record_initial = False
        config.validation.enabled = False
        p = api.prepare(config, training=True)
        scene = p.scenario
        buffers = tuple(
            replace(b, storage=replace(b.storage, capacity=10))
            if b.role == "system_input"
            else b
            for b in scene.factory.buffers
        )
        p = replace(
            p,
            scenario_json=canonical_json(
                replace(scene, factory=replace(scene.factory, buffers=buffers))
            ),
        )
        from smartsom.config.experiment_v3 import DQNParameters

        if algorithm == "dqn":
            parameters = DQNParameters(batch_size=2, warmup_ticks=0).model_dump()
            p = replace(p, parameters_json=canonical_json(parameters))
        with tempfile.TemporaryDirectory(prefix="physical-job-restore-") as temp:
            session = TrainingSession(p, Path(temp), {})
            session.save = lambda **kwargs: None
            try:
                session.step_update()
                boundary = copy.deepcopy(session.state_dict())
                session.step_update()
                expected = copy.deepcopy(session.state_dict())
                session.close()
                session = TrainingSession(p, Path(temp), {})
                session.save = lambda **kwargs: None
                session.restore(boundary)
                session.step_update()
                actual = session.state_dict()
                keys = (
                    "learners",
                    "policies",
                    "collector",
                    "replays",
                    "random",
                    "torch_random",
                    "numpy_random",
                    "python_random",
                    "target_clock",
                    "optimizations",
                    "ticks",
                    "updates",
                    "actions",
                )
                assert all(equal(expected[key], actual[key]) for key in keys), (
                    algorithm,
                    [key for key in keys if not equal(expected[key], actual[key])],
                )
                before = copy.deepcopy(actual)
                bad = copy.deepcopy(actual)
                group = next(
                    g
                    for g, policy in session.policies.items()
                    if hasattr(policy, "encoder")
                )
                bad["policies"][group]["encoder"]["physical_job_encoder"] = (
                    identity.replace("True", "False")
                )
                bad["ticks"] = 9999
                try:
                    session.restore(bad)
                except ValueError as exc:
                    assert "observation identity changed" in str(exc)
                else:
                    raise AssertionError("mismatched identity restored")
                assert all(
                    equal(before[key], session.state_dict()[key]) for key in keys
                )
                assert not list(Path(temp).rglob("*.pt")) and not list(
                    Path(temp).rglob("*.pkl")
                )
            finally:
                session.close()
    print("MATCHING_NATIVE_PPO_DQN_RESTORE_AND_PREFLIGHT_REJECTION_PASSED")


def disk_resume_probe():
    import hashlib
    import shutil

    from smartsom._filesystem import native_path
    from smartsom.config.experiment_v3 import model_location
    from smartsom.config.policies import ModelSelector
    from smartsom.experiments import composable as native
    from smartsom.experiments.evidence import write_json

    root = Path(os.environ["SMARTSOM_TEST_ROOT"])
    output = Path(os.environ["SMARTSOM_TEST_OUTPUT"])
    torch.set_num_threads(1)
    install(include_inspection=True)
    for algorithm in ("ppo", "dqn"):
        config = api.load_config(
            root / ("configs/test/runs/train_all_" + algorithm + ".yaml")
        )
        config.scenario = str(root / "configs/test/scenarios/small_matrix_auto.yaml")
        config.composition = str(
            root / ("configs/test/compositions/small_train_" + algorithm + ".yaml")
        )
        config.training.groups = ("machine", "buffer", "dispatcher")
        config.training.total_ticks = 16
        config.training.ticks_per_update = 8
        config.training.record_initial = False
        config.validation.enabled = False
        config.output.root = str(output / algorithm)
        config.output.name = "native-roundtrip"
        p = api.prepare(config, training=True)
        scene = p.scenario
        factory = replace(
            scene.factory,
            buffers=tuple(
                replace(b, storage=replace(b.storage, capacity=10))
                if b.role == "system_input"
                else b
                for b in scene.factory.buffers
            ),
        )
        p = replace(
            p,
            scenario_json=canonical_json(replace(scene, factory=factory)),
            origins_json=json.dumps({"note": "调度 café"}, ensure_ascii=False),
        )
        run, record, p = native.allocate(p, "training")
        session = native.TrainingSession(p, run, record)
        try:
            session.step_update()
            selected = native.checkpoint_path(run, "last")
            metadata_path = selected / "groups/dispatcher/model.json"
            metadata = json.loads(
                native_path(metadata_path).read_text(encoding="utf-8")
            )
            metadata["label"] = "调度 café"
            write_json(metadata_path, metadata)
            copied = output / ("copied-" + algorithm)
            shutil.copytree(native_path(run), native_path(copied))
            session.step_update()
            expected = copy.deepcopy(session.state_dict())
        finally:
            session.close()
        loaded = native.prepared_from_run(copied)
        assert json.loads(loaded.origins_json)["note"] == "调度 café"
        resolved = model_location(ModelSelector(source=str(copied), group="dispatcher"))
        assert resolved["metadata"]["label"] == "调度 café"
        api.resume(copied)
        last = native.checkpoint_path(copied, "last")
        with native_path(last / "continuation.pkl").open("rb") as stream:
            actual = pickle.load(stream)
        assert (
            actual["ticks"] == expected["ticks"]
            and actual["updates"] == expected["updates"]
        )
        assert actual["actions"] == expected["actions"]
        for group, saved in actual["policies"].items():
            assert saved.keys() == expected["policies"][group].keys()
            if "encoder" in saved:
                assert saved["encoder"] == expected["policies"][group]["encoder"]
            if "generator" in saved:
                assert torch.equal(
                    saved["generator"], expected["policies"][group]["generator"]
                )
        for group, policy_state in actual["learners"].items():
            # Native continuation lossless serialization is covered by strict RAM tests.
            assert policy_state.keys() == expected["learners"][group].keys()
        data = native_path(last / "continuation.pkl").read_bytes()
        native_path(last / "continuation.pkl").write_bytes(data + b"corrupt")
        try:
            api.resume(copied)
        except ValueError as exc:
            assert "continuation state hash mismatch" in str(exc)
        else:
            raise AssertionError("corrupted continuation resumed")
        assert (
            hashlib.sha256(data).hexdigest()
            != hashlib.sha256(data + b"corrupt").hexdigest()
        )
    print("REAL_NATIVE_LONG_SAVE_COPY_RESOLVE_RESUME_AND_HASH_GATE_PASSED")


if __name__ == "__main__":
    import sys

    if "--disk-resume" in sys.argv:
        disk_resume_probe()
    elif "--restore" in sys.argv:
        restore_probe()
    else:
        main()
