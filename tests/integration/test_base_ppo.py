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
    assert all(r["research_metrics"]["level"] == "detailed" for r in cases)
    from smartsom.telemetry.research_comparison import robustness

    normal, low = [r for r in cases if r["method"] == "ppo"][:2]
    assert robustness(normal, low, 0, min(normal["tick"], low["tick"]))["status"] in {
        "observed",
        "not_applicable",
    }
    episodes = [
        json.loads(line)
        for line in (tmp_path / "train/episodes.jsonl").read_text().splitlines()
    ]
    assert episodes and all(
        r["research_metrics"]["level"] == "summary" and r["ppo_update_seconds"] >= 0
        for r in episodes
    )
    before = cases
    evaluate(
        SimpleNamespace(
            experiment=path,
            output=tmp_path / "evaluation",
            calibration=calibration,
            method=["rule", "random", "ppo"],
            checkpoint=str(tmp_path / "train/last.pt"),
            selection="sample",
            split="validation",
            cases=4,
            action_seeds=1,
        )
    )
    assert [
        json.loads(line)
        for line in (tmp_path / "evaluation/cases.jsonl").read_text().splitlines()
    ] == before
    worlds = {
        method: {
            (r["scenario_seed"], r["condition"]) for r in cases if r["method"] == method
        }
        for method in ("rule", "random", "ppo")
    }
    assert worlds["rule"] == worlds["random"] == worlds["ppo"]
    assert (tmp_path / "evaluation" / "trace-ppo.jsonl").is_file()


def test_observation_checkpoint_versions_and_legacy_default(tmp_path):
    torch = pytest.importorskip("torch")
    from smartsom.learning.base_ppo import BasePPO

    options = BaseAlgorithm(schema="smartsom.base-algorithm/v1")
    original = BasePPO(options, 1)
    checkpoint = tmp_path / "model.pt"
    original.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    del payload["options"]["observation_schema"]
    torch.save(payload, checkpoint)
    restored = BasePPO(options, 2)
    restored.load(checkpoint)
    assert torch.equal(original.generator.get_state(), restored.generator.get_state())
    corrected = BasePPO(options.model_copy(update={"observation_schema": "native-v2"}))
    with pytest.raises(ValueError, match="contract mismatch"):
        corrected.load(checkpoint)


def test_mixed_policy_records_only_learning_stages_and_restores_rng(tmp_path):
    from smartsom.algorithms.base_policy import BoundBasePolicy

    options = BaseAlgorithm(
        schema="smartsom.base-algorithm/v1",
        method="rule",
        epochs=1,
        bindings={"dispatcher.pickup": "ppo", "pre.accept": "random"},
    )
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = BoundBasePolicy(options, 9)
    report, rewards, critic = episode(
        base_factory("small"),
        workload,
        options,
        policy,
        123,
        "normal",
        1000,
        max_ticks=100,
    )
    assert policy.records
    assert all(
        (r["request"].group, r["request"].stage) == ("dispatcher", "pickup")
        for r in policy.records
    )
    assert len(rewards) == report["tick"] == 100
    policy.update(rewards, terminal=False, final_critic=critic)
    assert set(policy.last_losses) == {"dispatcher"}
    checkpoint = tmp_path / "mixed.pt"
    policy.save(checkpoint)
    restored = BoundBasePolicy(options, 999)
    restored.load(checkpoint)
    assert restored._states() == policy._states()
    assert restored.updates == policy.updates > 0


def test_evaluation_policy_preserves_explicit_thread_allocation():
    import torch

    from smartsom.algorithms.base_policy import build_policy

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        options = BaseAlgorithm(
            schema="smartsom.base-algorithm/v1", bindings={"pre.accept": "rule"}
        )
        build_policy(options, training=False, numerical_threads=torch.get_num_threads())
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


def test_research_recording_preserves_real_ppo_update_and_restore(tmp_path):
    torch = pytest.importorskip("torch")
    from smartsom.learning.base_ppo import BasePPO

    options = BaseAlgorithm(schema="smartsom.base-algorithm/v1", epochs=1)
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    snapshots = []
    for level in ("off", "summary", "detailed"):
        policy = BasePPO(options, 31)
        row, rewards, critic = episode(
            base_factory("small"),
            workload,
            options,
            policy,
            71,
            "high",
            1000,
            max_ticks=80,
            metrics_level=level,
        )
        decisions = [(r["request"], r["critic"], r["action"]) for r in policy.records]
        policy.update(rewards, terminal=False, final_critic=critic)
        checkpoint = tmp_path / f"{level}.pt"
        policy.save(checkpoint)
        restored = BasePPO(options, 999)
        restored.load(checkpoint)
        assert restored.updates == policy.updates > 0
        snapshots.append(
            (decisions, rewards, torch.load(checkpoint, weights_only=False))
        )

    def equal(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(snapshots[0], snapshots[1])
    equal(snapshots[0], snapshots[2])
