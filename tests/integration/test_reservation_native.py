"""Public reservation execution and performance-preserving learning contracts."""

import copy
import json
from pathlib import Path

import pytest

from smartsom.config.base_diagnostic import BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options

ROOT = Path(__file__).resolve().parents[2]


def test_cached_pack_and_updates_are_exact(monkeypatch):
    torch = pytest.importorskip("torch")
    import smartsom.learning.ppo.learner as module

    options = algorithm_options(schema="smartsom.base-algorithm/v1", epochs=2)
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = module.BasePPO(options, 17)
    result, rewards, critic = episode(
        base_factory("small"),
        workload,
        options,
        policy,
        19,
        "normal",
        1000,
        max_ticks=200,
    )
    baseline = module.BasePPO(options, 17)
    baseline.models.load_state_dict(policy.models.state_dict())
    baseline.records = copy.deepcopy(policy.records)
    baseline.rng.setstate(policy.rng.getstate())
    records = policy.records[:16]
    requests, critics = [r["request"] for r in records], [r["critic"] for r in records]
    from smartsom.representations.base.batching import pack as original_pack

    a = original_pack(requests, critics, options.count_scale)
    b = original_pack(
        requests, critics, options.count_scale, module.encode_requests(requests)
    )
    for key in a:
        if key == "sets":
            for x, y in zip(a[key], b[key], strict=True):
                assert all(torch.equal(i, j) for i, j in zip(x, y, strict=True))
        elif torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key])
        else:
            assert a[key] == b[key]
    policy.update(rewards, terminal=False, final_critic=critic)
    baseline.pack = lambda requests, critics, count_scale, encoded=None, **kwargs: (
        original_pack(requests, critics, count_scale, **kwargs)
    )
    baseline.update(rewards, terminal=False, final_critic=critic)
    assert policy.updates == baseline.updates > 0
    assert policy.last_losses == baseline.last_losses
    for key, tensor in policy.models.state_dict().items():
        assert torch.equal(tensor, baseline.models.state_dict()[key]), key


def test_evaluation_skips_critic_and_preserves_actions(monkeypatch):
    torch = pytest.importorskip("torch")
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
    from smartsom.learning.ppo.learner import BasePPO
    from smartsom.representations.base.batching import pack

    request = BaseDecision(
        0,
        "pre",
        "pre_1",
        "accept",
        (
            BaseCandidate("a", ("accept", "a"), vector(1)),
            BaseCandidate("stop", ("STOP",), vector()),
        ),
        vector(),
        ((), (), (), ()),
    )
    options = algorithm_options(schema="smartsom.base-algorithm/v1")
    for selection in ("sample", "argmax"):
        policy = BasePPO(options, 7, training=False, selection=selection)
        batch = pack([request], [vector()], options.count_scale)
        with torch.no_grad():
            logits, _ = policy.models["pre"](batch)
            assert torch.equal(logits, policy.models["pre"].actor(batch))
            index = (
                torch.multinomial(
                    logits.softmax(-1), 1, generator=torch.Generator().manual_seed(7)
                )
                if selection == "sample"
                else logits.argmax(-1)
            )

        def forbidden(*args, **kwargs):
            raise AssertionError("evaluation invoked critic")

        monkeypatch.setattr(policy.models["pre"].critic, "forward", forbidden)
        assert policy.choose(request, None) == request.candidates[index.item()].action


def test_native_public_run_freezes_inputs_and_smokes_all_methods(tmp_path):
    pytest.importorskip("torch")
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation import run

    path = ROOT / "experiments/examples/small/default/experiment.yaml"
    overrides = [
        f'experiment.run.output="{tmp_path / "run"}"',
        "experiment.run.smoke_ticks=40",
        "algorithm.production.policies.ppo.learner.params.epochs=1",
    ]
    summary = compile_reservation(path, sets=overrides).summary()
    assert summary["machines"] == 8 and summary["orders"] == 128
    assert not (tmp_path / "run").exists()
    plan = compile_reservation(path, sets=overrides)
    result = run(plan, display_options={"progress": "off", "verbose": False})
    assert result["status"] == "completed"
    rows = json.loads((tmp_path / "run/summary.json").read_text())["results"]
    assert {r["method"] for r in rows} == {"rule", "random", "ppo"}
    assert all(r["tick"] == 40 for r in rows)
    assert rows[-1]["updates"] > 0
    assert (tmp_path / "run/last.pt").is_file()
    assert (tmp_path / "run/inputs/factory.yaml").is_file()
    with pytest.raises(ValueError, match="already exists"):
        run(plan, display_options={"progress": "off", "verbose": False})


def test_native_calibration_train_and_evaluate_share_scenario_identity(tmp_path):
    pytest.importorskip("torch")
    from smartsom import api
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    path = compile_reservation(
        ROOT / "experiments/examples/small/default/experiment.yaml",
        sets=["experiment.tracking.mode=disabled"],
    ).materialize(inputs)
    shared = [
        "workload.orders=8",
        "workload.initial_orders=4",
        "workload.mean_interarrival=4",
        "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        "algorithm.production.policies.ppo.learner.params.epochs=1",
        "experiment.validation_cases=4",
        "experiment.test_cases=4",
        "experiment.run.seconds=120",
    ]

    def execute(task, extra):
        return api.run_experiment(
            path,
            sets=shared
            + [
                f"experiment.run.task={task}",
                f'experiment.run.output="{tmp_path / task}"',
            ]
            + extra,
        )

    execute("calibrate", ["experiment.run.calibration_cases=1"])
    proof = tmp_path / "calibrate/calibration.json"
    assert json.loads(proof.read_text())["status"] == "frozen"
    calibrated = [f'experiment.run.calibration="{proof}"']
    result = execute("train", calibrated + ["experiment.train_ticks=80"])
    assert result["physical_ticks"] == 80 and result["updates"] > 0
    execute(
        "evaluate",
        calibrated
        + [
            f'experiment.run.checkpoint="{tmp_path / "train/last.pt"}"',
            "experiment.run.selection=argmax",
        ],
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluate/cases.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 12
    assert all(row["selection"] == "argmax" for row in rows)
    assert len({row["scenario_seed"] for row in rows}) == 1
    summary = json.loads((tmp_path / "evaluate/summary.json").read_text())
    for method in summary.values():
        if method["all"]["not_drained_rate"]:
            assert method["all"]["passing_rate"] is None


def test_diagnostics_off_never_constructs_counters_and_rules_skip_critic(monkeypatch):
    from smartsom.engine.base_protocol import BaseProtocol
    from smartsom.experiments import reservation_diagnostics
    from smartsom.policies.rule import BaseRule

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled diagnostic/critic path was invoked")

    monkeypatch.setattr(reservation_diagnostics, "DecisionCounters", forbidden)
    monkeypatch.setattr(BaseProtocol, "critic_features", forbidden)
    options = algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
    )
    result, rewards, critic = episode(
        base_factory("small"),
        workload,
        options,
        BaseRule(),
        1,
        "normal",
        100,
        max_ticks=10,
    )
    assert critic is None and "diagnostics" not in result
    assert len(rewards) == 10
