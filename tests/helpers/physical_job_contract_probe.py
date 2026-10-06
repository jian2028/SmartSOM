import copy
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


if __name__ == "__main__":
    import sys

    if "--restore" in sys.argv:
        restore_probe()
    else:
        main()
