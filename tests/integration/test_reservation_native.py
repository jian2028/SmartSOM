"""Public reservation execution and performance-preserving learning contracts."""

import copy
import json
from pathlib import Path

import pytest

from smartsom.config.base_diagnostic import BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options
from tests.helpers.observations import numeric_candidate, numeric_request

ROOT = Path(__file__).resolve().parents[2]


def test_cached_pack_and_updates_are_exact(monkeypatch):
    torch = pytest.importorskip("torch")
    import smartsom.learning.ppo.learner as module

    options = algorithm_options(epochs=2)
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
    from smartsom.domain.observation_data import vector
    from smartsom.learning.ppo.learner import BasePPO
    from smartsom.representations.base.batching import pack

    request = numeric_request(
        0,
        "pre",
        "pre_1",
        "accept",
        (
            numeric_candidate("a", ("accept", "a"), vector(1)),
            numeric_candidate("stop", ("STOP",), vector()),
        ),
        vector(),
        ((), (), (), ()),
    )
    options = algorithm_options()
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
        "scenario.jobs.orders.total=8",
        "scenario.jobs.orders.initial=4",
        "scenario.jobs.arrivals.mean_interval_ticks=4",
        "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
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
    from smartsom.experiments import reservation_diagnostics
    from smartsom.policies.rule import BaseRule

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled diagnostic/critic path was invoked")

    monkeypatch.setattr(reservation_diagnostics, "DecisionCounters", forbidden)
    monkeypatch.setattr(
        "smartsom.algorithms.training.TrainingComponents.value_input", forbidden
    )
    options = algorithm_options(method="rule")
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


@pytest.mark.parametrize("social", [False, True])
@pytest.mark.parametrize("empty_sets", [False, True])
def test_single_request_pack_matches_general_path_exactly(social, empty_sets):
    import torch

    from smartsom.domain.observation_data import vector
    from smartsom.domain.social import SocialCandidate, SocialLink
    from smartsom.representations.base.batching import encode_requests, pack

    candidates = (
        numeric_candidate("job", ("job",), vector(1.25, -3)),
        numeric_candidate("stop", ("STOP",), vector()),
        numeric_candidate("wait", ("EMPTY_WAIT",), vector()),
    )
    request = numeric_request(
        0,
        "dispatcher",
        "v",
        "pickup",
        candidates,
        vector(0.125),
        ((), (), (), ())
        if empty_sets
        else ((vector(1), vector(2)), (), (vector(3),), (vector(4),)),
        social=tuple(
            SocialCandidate(
                c.identity, (SocialLink("peer", "station", vector(2), (vector(3),)),)
            )
            for c in candidates
        )
        if social
        else None,
    )

    def equal(a, b):
        if torch.is_tensor(a):
            assert a.dtype == b.dtype and a.shape == b.shape
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

    for dtype in (torch.float32, torch.float64):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            fast = pack([request], None, 20)
            general = pack([request], None, 20, encode_requests([request]))
            equal(fast, general)
            # A later call owns fresh tensor storage; no retained input is overwritten.
            again = pack([request], None, 20)
            again["candidates"].zero_()
            equal(fast, general)
        finally:
            torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "training,selection", [(False, "argmax"), (False, "sample"), (True, "argmax")]
)
def test_dqn_singleton_preserves_action_rng_and_training_records(
    training, selection, monkeypatch
):
    from unittest.mock import Mock

    torch = pytest.importorskip("torch")
    from smartsom.domain.observation_data import vector
    from smartsom.learning.dqn.learner import BaseDQN

    options = algorithm_options(method="dqn")
    request = numeric_request(
        0,
        "pre",
        "pre_1",
        "accept",
        (numeric_candidate("only", ("agv_1",), vector(1)),),
        vector(),
        ((), (), (), ()),
    )
    policy = BaseDQN(options, 7, training=training, selection=selection)
    reference = BaseDQN(options, 7, training=training, selection=selection)
    # A custom packer requires the ordinary forward path, providing a reference
    # without adding a production switch that could change scientific inputs.
    prepare = reference.pack
    reference.pack = lambda *a, **kw: prepare(*a, **kw)
    expected = reference.choose(request)
    global_rng = torch.get_rng_state().clone()
    context = Mock(wraps=torch.no_grad)
    monkeypatch.setattr(torch, "no_grad", context)
    assert policy.choose(request) == expected == ("agv_1",)
    assert torch.equal(policy.generator.get_state(), reference.generator.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert list(policy.records) == list(reference.records)
    assert policy.decisions == reference.decisions == int(training)
    assert context.call_count == (0 if not training and selection == "argmax" else 1)


@pytest.mark.parametrize(
    "customization", ["hook", "global_hook", "actor", "submodule", "packer", "learner"]
)
def test_dqn_singleton_preserves_custom_forward_effects(customization):
    torch = pytest.importorskip("torch")
    from smartsom.domain.observation_data import vector
    from smartsom.learning.dqn.learner import BaseDQN

    calls = []

    class CustomLearner(BaseDQN):
        def _pack(self, requests):
            calls.append(True)
            return super()._pack(requests)

    cls = CustomLearner if customization == "learner" else BaseDQN
    policy = cls(algorithm_options(method="dqn"), 9, training=False, selection="argmax")
    request = numeric_request(
        0,
        "pre",
        "pre_1",
        "accept",
        (numeric_candidate("only", ("a",), vector()),),
        vector(),
        ((), (), (), ()),
    )
    model = policy.models["pre"]
    handle = None
    if customization == "hook":
        handle = model.encoder.register_forward_hook(lambda *a: calls.append(True))
    elif customization == "global_hook":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda *a: calls.append(True)
        )
    elif customization == "actor":
        original = model.actor

        def actor(batch):
            calls.append(True)
            return original(batch)

        model.actor = actor
    elif customization == "submodule":

        class ObservedLinear(torch.nn.Linear):
            def forward(self, x):
                calls.append(True)
                return super().forward(x)

        original = model.encoder.candidate[0]
        changed = ObservedLinear(original.in_features, original.out_features)
        changed.load_state_dict(original.state_dict())
        model.encoder.candidate[0] = changed
    elif customization == "packer":
        prepare = policy.pack

        def pack(*a, **kw):
            calls.append(True)
            return prepare(*a, **kw)

        policy.pack = pack
    try:
        assert policy.choose(request) == ("a",)
        assert calls
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize(
    "group,stage",
    [
        ("machine", "process"),
        ("dispatcher", "pickup"),
        ("dispatcher", "delivery"),
        ("pre", "accept"),
        ("pre", "service"),
        ("post", "accept"),
        ("post", "service"),
        ("input", "accept"),
        ("input", "service"),
    ],
)
def test_social_singleton_matches_forward_and_preserves_validation(group, stage):
    from dataclasses import replace

    import torch

    from smartsom.domain.observation_data import vector
    from smartsom.domain.social import SocialCandidate, SocialLink
    from smartsom.learning.dqn.learner import BaseDQN

    options = algorithm_options(method="dqn", social_arm="all")
    actual = BaseDQN(options, 11, training=False, selection="argmax")
    reference = BaseDQN(options, 11, training=False, selection="argmax")
    pack = reference.pack
    reference.pack = lambda *a, **kw: pack(*a, **kw)
    link = SocialLink("peer", "station", vector(), (vector(1),), False, 1)
    request = numeric_request(
        0,
        group,
        "owner",
        stage,
        (numeric_candidate("only", ("action",), vector()),),
        vector(),
        ((), (), (), ()),
        social=(SocialCandidate("only", (link,)),),
    )
    rng = torch.get_rng_state().clone()
    assert actual.choose(request) == reference.choose(request) == ("action",)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(actual.generator.get_state(), reference.generator.get_state())
    assert not actual.records and actual.decisions == 0
    for bad in [
        replace(request, inputs=replace(request.inputs, social=None)),
        replace(
            request,
            inputs=replace(request.inputs, social=(SocialCandidate("wrong", (link,)),)),
        ),
    ]:
        for policy in (actual, reference):
            with pytest.raises(ValueError):
                policy.choose(bad)
    calls = []
    handle = actual.models[group].head.social.register_forward_hook(
        lambda *a: calls.append(True)
    )
    try:
        assert actual.choose(request) == ("action",)
        assert calls
    finally:
        handle.remove()


def test_social_singleton_retains_base_shape_and_inactive_role_errors():
    from dataclasses import replace

    from smartsom.domain.observation_data import vector
    from smartsom.domain.social import SocialCandidate
    from smartsom.learning.dqn.learner import BaseDQN

    for arm in ("all", "machine"):
        options = algorithm_options(method="dqn", social_arm=arm)
        actual = BaseDQN(options, 7, training=False, selection="argmax")
        reference = BaseDQN(options, 7, training=False, selection="argmax")
        pack = reference.pack
        reference.pack = lambda *a, **kw: pack(*a, **kw)
        request = numeric_request(
            0,
            "dispatcher",
            "v",
            "pickup",
            (numeric_candidate("only", ("input",), vector()),),
            vector(),
            ((), (), (), ()),
            social=(SocialCandidate("only", ()),) if arm == "all" else None,
        )
        bad_rows = [
            replace(request, inputs=replace(request.inputs, context=(0.0,))),
            replace(request, inputs=replace(request.inputs, candidates=((0.0,),))),
            replace(request, inputs=replace(request.inputs, sets=((), (), ()))),
            replace(
                request, inputs=replace(request.inputs, sets=(((0.0,),), (), (), ()))
            ),
            replace(
                request,
                inputs=replace(request.inputs, social=(SocialCandidate("wrong", ()),)),
            ),
        ]
        for bad in bad_rows:
            with pytest.raises((ValueError, RuntimeError)) as expected:
                reference.choose(bad)
            with pytest.raises(type(expected.value)) as observed:
                actual.choose(bad)
            assert str(observed.value) == str(expected.value)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_social_packing_preserves_ragged_metadata_and_detachment(dtype_name):
    from dataclasses import replace

    import torch

    from smartsom.domain.observation_data import vector
    from smartsom.domain.social import SocialCandidate, SocialLink
    from smartsom.learning.social_actor import pack_social

    previous = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, dtype_name))
    try:
        links = (
            SocialLink("a", "station", vector(2), (vector(3), vector(4)), True, 3),
            SocialLink("b", "vehicle", vector(5), (), False, 0),
        )
        request = numeric_request(
            0,
            "pre",
            "owner",
            "accept",
            (
                numeric_candidate("one", ("a",), vector()),
                numeric_candidate("two", ("b",), vector()),
            ),
            vector(),
            ((), (), (), ()),
            social=(SocialCandidate("two", (links[1],)), SocialCandidate("one", links)),
        )
        packed = pack_social(
            [request, replace(request, inputs=replace(request.inputs, social=None))],
            3,
            7,
        )
        assert packed["query"].dtype == getattr(torch, dtype_name)
        assert packed["meta"].dtype == packed["query"].dtype
        assert packed["indices"].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
        expected = torch.tensor([[0, 1, 0, 1, 3 / 7], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        assert torch.equal(packed["meta"], expected)
        assert packed["peer_mask"].tolist() == [
            [True, True],
            [False, False],
            [False, False],
        ]
        assert packed["link_mask"].shape == (2, 3, 2)
        assert packed["link_mask"].sum() == 3
        assert not packed["link_mask"][1].any()
        assert packed["query"][:, 0].tolist() == [2, 5, 5]
        packed["meta"].fill_(9)
        assert torch.equal(pack_social([request], 3, 7)["meta"], expected)
        empty = pack_social(
            [replace(request, inputs=replace(request.inputs, social=None))], 3, 7
        )
        assert empty["meta"].shape == (0, 5)
        assert empty["indices"].shape == (0, 3)
        assert empty["peers"].shape == (0, 1, 48)
        bad = replace(
            request,
            inputs=replace(
                request.inputs,
                social=(
                    SocialCandidate("one", (replace(links[0], query=(1, 2)),)),
                    SocialCandidate("two", ()),
                ),
            ),
        )
        with pytest.raises(RuntimeError):
            pack_social([bad], 3, 7)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("placement", ["root", "nested", "class"])
def test_singleton_retains_custom_module_enumeration(monkeypatch, placement):
    from smartsom.domain.observation_data import vector
    from smartsom.learning.dqn.learner import BaseDQN

    policy = BaseDQN(
        algorithm_options(method="dqn"), 11, training=False, selection="argmax"
    )
    model = policy.models["pre"]
    layer = model if placement == "root" else model.encoder
    calls = []
    if placement == "class":
        original = type(layer).named_modules

        def named_modules(self, *args, **kwargs):
            calls.append(True)
            yield from original(self, *args, **kwargs)

        monkeypatch.setattr(type(layer), "named_modules", named_modules)
    else:
        original = layer.named_modules

        def named_modules(*args, **kwargs):
            calls.append(True)
            yield from original(*args, **kwargs)

        monkeypatch.setattr(layer, "named_modules", named_modules)
    request = numeric_request(
        0,
        "pre",
        "owner",
        "accept",
        (numeric_candidate("only", ("action",), vector()),),
        vector(),
        ((), (), (), ()),
    )
    assert policy.choose(request) == ("action",)
    assert calls
