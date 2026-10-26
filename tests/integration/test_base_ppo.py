"""Real optional Torch updates, permutation invariance and checkpoint restore."""

import pytest

from smartsom.config.base_diagnostic import BaseAlgorithm, BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.experiments.base_diagnostic import episode


def test_base_ppo_updates_all_groups_and_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    from smartsom.learning.base_ppo import BasePPO, pack

    options = BaseAlgorithm(
        schema="smartsom.base-algorithm/v1", epochs=1, batch_size=128
    )
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = BasePPO(options, 1)
    initial = {k: v.clone() for k, v in policy.models.state_dict().items()}
    report, rewards, critic = episode(
        base_factory("small"), workload, options, policy, 123, "normal", 1000
    )
    record = next(r for r in policy.records if len(r["request"].candidates) > 1)
    request = record["request"]
    from dataclasses import replace

    reversed_request = replace(
        request,
        candidates=tuple(reversed(request.candidates)),
        sets=tuple(tuple(reversed(rows)) for rows in request.sets),
    )
    model = policy.models[request.group]
    with torch.no_grad():
        x, _ = model(pack([request], [record["critic"]], options.count_scale))
        y, _ = model(pack([reversed_request], [record["critic"]], options.count_scale))
    assert torch.allclose(x, y.flip(1), atol=1e-6)
    losses = policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    assert set(losses) == {"machine", "dispatcher", "pre", "post", "input"}
    for group in losses:
        assert any(
            not torch.equal(initial[k], v)
            for k, v in policy.models.state_dict().items()
            if k.startswith(group + ".")
        )
    policy.save(tmp_path / "last.pt")
    restored = BasePPO(options, 999)
    restored.load(tmp_path / "last.pt")
    assert restored.updates == policy.updates > 0
    for k, v in policy.models.state_dict().items():
        assert torch.equal(v, restored.models.state_dict()[k])
    assert restored.choose(request, record["critic"]) == policy.choose(
        request, record["critic"]
    )


def test_four_file_calibrate_train_and_paired_evaluate(tmp_path):
    pytest.importorskip("torch")
    import json
    from types import SimpleNamespace

    import yaml

    from smartsom.config.base_diagnostic import BaseExperiment
    from smartsom.config.factory_design import save_factory_design
    from smartsom.experiments.base_diagnostic import calibrate, evaluate, train

    save_factory_design(tmp_path / "factory.yaml", base_factory("small"))
    w = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    a = BaseAlgorithm(schema="smartsom.base-algorithm/v1", epochs=1)
    ex = BaseExperiment(
        schema="smartsom.base-experiment/v1",
        factory="factory.yaml",
        workload="workload.yaml",
        algorithm="algorithm.yaml",
        validation_cases=4,
        test_cases=4,
        record_first_case=True,
    )
    for name, value in (("workload", w), ("algorithm", a), ("experiment", ex)):
        (tmp_path / f"{name}.yaml").write_text(
            yaml.safe_dump(value.model_dump(mode="json", by_alias=True))
        )
    path = tmp_path / "experiment.yaml"
    calibration = tmp_path / "calibration" / "calibration.json"
    calibrate(
        SimpleNamespace(
            experiment=path,
            output=calibration.parent,
            seconds=120,
            cases=1,
            max_drain=4000,
        )
    )
    assert json.loads(calibration.read_text())["status"] == "frozen"
    train(
        SimpleNamespace(
            experiment=path,
            output=tmp_path / "train",
            calibration=calibration,
            seed=7,
            ticks=300,
            seconds=120,
        )
    )
    progress = json.loads((tmp_path / "train" / "progress.json").read_text())
    assert progress["physical_ticks"] == 300 and progress["updates"] > 0
    evaluate(
        SimpleNamespace(
            experiment=path,
            output=tmp_path / "evaluation",
            calibration=calibration,
            method=["rule", "random", "ppo"],
            checkpoint=str(tmp_path / "train" / "last.pt"),
            selection="sample",
            split="validation",
            cases=4,
            action_seeds=1,
        )
    )
    cases = [
        json.loads(row)
        for row in (tmp_path / "evaluation" / "cases.jsonl").read_text().splitlines()
    ]
    assert len(cases) == 12
    worlds = {
        method: {
            (r["scenario_seed"], r["condition"]) for r in cases if r["method"] == method
        }
        for method in ("rule", "random", "ppo")
    }
    assert worlds["rule"] == worlds["random"] == worlds["ppo"]
    assert (tmp_path / "evaluation" / "trace-ppo.jsonl").is_file()


def test_empty_wait_head_and_scaled_update_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from smartsom.learning.base_ppo import BasePPO, pack

    options = BaseAlgorithm(
        schema="smartsom.base-algorithm/v1",
        empty_wait=True,
        reconstructed_inputs=True,
        epochs=1,
    )
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=4,
        initial_orders=2,
        mean_interarrival=2,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = BasePPO(options, 71)
    report, rewards, critic = episode(
        base_factory("small"), workload, options, policy, 81, "normal", 30
    )
    row = next(
        r
        for r in policy.records
        if any(c.action == ("EMPTY_WAIT",) for c in r["request"].candidates)
    )
    inputs = pack([row["request"]], [row["critic"]], options.count_scale)
    logits, _ = policy.models["dispatcher"](inputs)
    assert torch.isfinite(logits[inputs["wait"]]).all()
    policy.update(
        [r / 1000 for r in rewards],
        terminal=report["status"] == "completed",
        final_critic=critic,
    )
    assert policy.updates > 0 and not policy.records
    policy.save(tmp_path / "last.pt")
    other = BasePPO(options, 99)
    other.load(tmp_path / "last.pt")
    assert other.choose(row["request"], row["critic"]) == policy.choose(
        row["request"], row["critic"]
    )


def test_local_chunk_resume_matches_continuous_training(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import importlib.util
    import json
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/local_base/run.py"
    spec = importlib.util.spec_from_file_location("local_base_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    options = BaseAlgorithm(
        schema="smartsom.base-algorithm/v1",
        empty_wait=True,
        reconstructed_inputs=True,
        epochs=1,
    )
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=4,
        initial_orders=2,
        mean_interarrival=2,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    monkeypatch.setattr(
        runner, "inputs", lambda arm: (base_factory("small"), workload, options)
    )
    monkeypatch.setattr(
        runner,
        "verify",
        lambda root: {
            "identity": "resume-test",
            "training_cutoff_unix": time.time() + 300,
        },
    )
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    runner.train(continuous, "C", 97, 2)
    runner.train(resumed, "C", 97, 1)
    runner.train(resumed, "C", 97, 2)
    a = torch.load(continuous / "train/C-97/last.pt", weights_only=False)
    b = torch.load(resumed / "train/C-97/last.pt", weights_only=False)
    assert a["updates"] == b["updates"] > 0
    assert a["local_state"]["physical_ticks"] == b["local_state"]["physical_ticks"]
    assert torch.equal(a["generator"], b["generator"])
    assert a["rng"] == b["rng"]
    for k in a["models"]:
        assert torch.equal(a["models"][k], b["models"][k]), k
    assert (
        json.loads((resumed / "train/C-97/progress.json").read_text())["episodes"] == 2
    )


def test_local_four_arm_initialization_and_rule_masks():
    torch = pytest.importorskip("torch")
    from smartsom.learning.base_ppo import BasePPO

    policies = [
        BasePPO(
            BaseAlgorithm(
                schema="smartsom.base-algorithm/v1",
                reservation_stop=stop,
                empty_wait=wait,
                reconstructed_inputs=True,
            ),
            17,
        )
        for stop, wait in ((True, False), (False, False), (True, True), (False, True))
    ]
    for policy in policies[1:]:
        assert all(
            torch.equal(v, policy.models.state_dict()[k])
            for k, v in policies[0].models.state_dict().items()
        )
