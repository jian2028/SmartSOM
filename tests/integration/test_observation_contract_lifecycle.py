"""Eight-tick public-contract lifecycles, with real learners and spawned readers."""

import copy
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml
from test_author_batch_cli_learning import _allocated, _command, _finish, _wait
from test_author_learning import ROOT, _require_cpu, _write
from test_configurable_batch import inputs

from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.experiment_v3 import PreparedComposition, model_location
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.policies import ModelSelector
from smartsom.config.production import named_seed
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import author_batch, author_driver, composable
from smartsom.learning.production_inference import build_groups, read_package

pytestmark = pytest.mark.learning


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source(directory, algorithm, *, native=False):
    experiments = inputs(directory, algorithm)
    method_path = directory / "algorithm.yaml"
    method = yaml.safe_load(method_path.read_text())

    def component(name, normalization, visibility):
        return dict(
            name=name,
            normalization=normalization,
            aggregation="per_entity",
            visibility=visibility,
        )

    parameters = dict(
        rounding="ceil",
        include_inspection=True,
        context=[
            component("decision", "projection", "actor"),
            component("self_target", "raw", "actor"),
            component("inventory", "capacity", "both"),
            component("work_summary", "projection", "critic"),
        ],
        candidates=[
            component(name, normalization, "actor")
            for name, normalization in (
                ("job", "projection"),
                ("semantics", "projection"),
                ("machine_mode", "projection"),
                ("dispatch_target", "capacity"),
                ("cargo", "projection"),
                ("movement", "projection"),
            )
        ],
    )
    for role in ("machine", "buffer", "dispatcher"):
        method["agents"][role]["default"]["extensions"].update(
            observation=dict(
                name="builtin.public_fields", version="1", parameters=parameters
            ),
            network_implementation=dict(name="builtin.scoped_candidate", version="1"),
        )
    _write(method_path, method)
    source = experiments / "experiment.yaml"
    document = yaml.safe_load(source.read_text())
    document["validation"] = dict(enabled=False)
    document["evaluation"].pop("no_eligible_best")
    document["evaluation"].update(checkpoint="last", initial_control=True)
    document["runtime"]["environment"]["processing_rounding"] = "ceil"
    document["execution"]["max_concurrent"] = 1
    if native:
        document["execution"].update(executor="native", tuning="off")
        document["runtime"]["sampling_processes"] = 1
    _write(source, document)
    return source, document


def _reconstruct_and_sample(serialized, package):
    """Called identically in the parent and a clean spawn interpreter."""
    prepared = PreparedComposition(**serialized)
    declarations = json.loads(prepared.policies_json)
    declaration = declarations["dispatcher"]
    declaration["implementation"] = dict(kind="model", model=dict(source=package))
    declaration["resolved_model"] = model_location(ModelSelector(source=package))
    prepared = replace(prepared, policies_json=canonical_json(declarations))
    policies, learners = build_groups(prepared, training=False)
    assert not learners
    policy = policies["dispatcher"]
    assert type(policy.encoder).__name__ == "PublicFieldsEncoder"
    assert type(policy.network).__name__ == "ScopedCandidateNetwork"
    identity = policy.encoder.state_dict()
    fingerprint = policy.fingerprint()
    simulation = ProductionSimulator(prepared.scenario, contract="v3")
    request = next(r for r in simulation.protocol.begin() if r.role == "dispatcher")
    choice = policy.choose(request)
    sampled = composable._worker_tick(
        ProductionSimulator(prepared.scenario, contract="v3"),
        pickle.dumps(policies, protocol=5),
        composable.bindings(prepared),
        "first_arrival",
        0,
    )
    return (
        identity,
        fingerprint,
        choice.action,
        choice.model_input,
        digest(sampled[0].snapshot()),
    )


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_real_native_initial_control_and_spawn_package_reconstruction(
    tmp_path, algorithm
):
    _require_cpu()
    source, _ = _source(tmp_path / "inputs", algorithm, native=True)
    plan = compile_experiment(source, require_dependencies=True)
    root = author_driver.allocate(plan)
    result = author_driver.execute_saved(root)
    assert result["status"] == "completed", result
    ledger_path = root / "entries/entry-0001/stages.json"
    stages = _json(ledger_path)["stages"]
    assert set(stages) == {"training", "evaluation", "initial_control"}
    training = Path(stages["training"]["run_dir"])
    assert _json(training / "run.json")["physical_ticks"] == 8
    assert any(_json(training / "run.json")["changed_weights"].values())
    initial = _json(Path(stages["initial_control"]["run_dir"]) / "run.json")
    primary = _json(Path(stages["evaluation"]["run_dir"]) / "run.json")
    assert initial["purpose"] == "initial_control"
    assert Path(initial["checkpoint"]) == composable.checkpoint_path(
        training, "initial"
    )
    assert len(initial["results"]) == 1 and not initial["summary"]["exceptions"]
    assert initial["results"][0]["seed"] == primary["results"][0]["seed"]
    assert initial["results"][0]["evaluation_label"] == "INITIAL-CONTROL"
    assert initial["results"][0]["ranking_eligible"] is False
    assert len(stages["initial_control"]["attempts"]) == 1
    before = ledger_path.read_bytes()
    assert author_driver.execute_saved(root)["status"] == "completed"
    assert ledger_path.read_bytes() == before

    package = composable.export(
        training, tmp_path / "dispatcher.zip", group="dispatcher"
    )
    metadata, _, encoder_state = read_package(package)
    frozen = asdict(composable.prepared_from_run(training))
    expected = _reconstruct_and_sample(frozen, str(package))
    assert expected[0] == encoder_state
    assert metadata["physical_job_encoder"] == expected[0]["public_fields"]
    with ProcessPoolExecutor(
        1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=composable._sampling_threads,
        initargs=(1,),
    ) as pool:
        actual = pool.submit(_reconstruct_and_sample, frozen, str(package)).result(
            timeout=90
        )
    assert actual == expected


def _frozen_source(tmp_path, algorithm):
    source, document = _source(tmp_path / "inputs", algorithm, native=True)
    scene = primitive(compile_experiment(source).entries[0].prepared.scenario)
    refs = {}
    for phase, seed in (
        ("training", 101),
        ("validation", 303001),
        ("evaluation", 404001),
    ):
        snapshot = copy.deepcopy(scene)
        snapshot["seed"] = seed
        # Complete short engineering windows, not a formal scientific budget.
        snapshot["tick_limit"] = 4
        snapshot["mode"] = "dynamic"
        target = tmp_path / (phase + ".json")
        target.write_text(
            canonical_json(
                dict(schema="smartsom.frozen-production-world/v1", scenario=snapshot)
            )
        )
        refs[phase] = dict(world=str(target), data_seed=seed + 7, environment_seed=seed)
        if phase != "training":
            refs[phase].update(id=phase + "_001", policy_seed=seed + 200000)
    document.pop("workload")
    document.pop("data_seed", None)
    document["runtime"].pop("environment")
    document["runtime"]["sampling_processes"] = 0
    document["training"].update(max_ticks=4)
    document["validation"] = dict(
        enabled=True,
        every_updates=None,
        updates=[1, 2],
        full_replay=False,
        best_mode="custom",
        metric="return",
        direction="max",
        failure_policy="ineligible",
    )
    document["evaluation"].update(
        checkpoint="best", no_eligible_best=dict(outcome="record_unselected")
    )
    document["evaluation"].pop("replications", None)
    document["training_environment_seeds"] = "legacy_policy"
    document["frozen_inputs"] = dict(
        schema="smartsom.frozen-experiment-inputs/v1",
        training=refs["training"],
        validation=[refs["validation"]],
        evaluation=[refs["evaluation"]],
    )
    _write(source, document)
    return source, document


def _restore_frozen_state(serialized, encoded_state, directory):
    prepared = PreparedComposition(**serialized)
    Path(directory).mkdir(parents=True, exist_ok=True)
    session = composable.TrainingSession(prepared, Path(directory), {})
    try:
        session.restore(pickle.loads(encoded_state))
        return pickle.dumps(session.state_dict(), protocol=5)
    finally:
        session.close()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_real_frozen_world_cases_and_full_restore_identity(
    tmp_path, monkeypatch, algorithm
):
    _require_cpu()
    source, document = _frozen_source(tmp_path, algorithm)
    prepared = compile_experiment(source, require_dependencies=True).entries[0].prepared
    root, record, archived = composable.allocate(prepared, "training")
    session = composable.TrainingSession(archived, root, record)
    try:
        assert session.sims[0].scenario.seed == named_seed(101, "training:0:0")
        session.step_update()
        state = session.state_dict()
        assert state["ticks"] == 4
        # Actual simulator, optimizer, collector and replay/RNG state survives
        # a new interpreter; no live learner object is passed across spawn.
        with ProcessPoolExecutor(
            1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=composable._sampling_threads,
            initargs=(1,),
        ) as pool:
            restored = pool.submit(
                _restore_frozen_state,
                asdict(archived),
                pickle.dumps(state, protocol=5),
                str(tmp_path / "restored"),
            ).result(timeout=90)
        from test_tuning_native_continuation import assert_state_equal

        assert_state_equal(state, pickle.loads(restored), exact=True)
        before = session.state_dict()
        for change in ("policy_root", "domain", "world"):
            modified = copy.deepcopy(document)
            if change == "policy_root":
                modified["frozen_inputs"]["evaluation"][0]["policy_seed"] += 1
            elif change == "domain":
                modified["training_environment_seeds"] = "policy"
            else:
                target = Path(modified["frozen_inputs"]["evaluation"][0]["world"])
                changed = json.loads(target.read_text())
                changed["provenance"] = {"revision": 2}
                target.write_text(canonical_json(changed))
            _write(source, modified)
            changed = compile_experiment(source).entries[0].prepared
            session.prepared = changed
            with pytest.raises(ValueError, match="identity changed"):
                session.restore(state)
            session.prepared = archived
            assert_state_equal(before, session.state_dict(), exact=True)
        # Real frozen-case evaluation applies the direct world and policy root.
        evaluation = composable.evaluation_recipe(
            archived, composable.checkpoint_path(root, "last")
        )
        cases = json.loads(evaluation.evaluation_json)
        actual_worlds, actual_roots, progress = [], [], []
        original_simulator, original_policies = (
            composable.ProductionSimulator,
            composable.policies_for,
        )
        import random

        import torch

        from smartsom.telemetry import runtime

        def simulator(scenario, **kwargs):
            actual_worlds.append(scenario.seed)
            return original_simulator(scenario, **kwargs)

        def policies(prepared, training=False, **kwargs):
            result = original_policies(prepared, training=training, **kwargs)
            actual_roots.append(kwargs["policy_seed"])
            for group, policy in result[0].items():
                seed = named_seed(kwargs["policy_seed"], "policy:" + group)
                if hasattr(policy, "generator"):
                    assert torch.equal(
                        policy.generator.get_state(),
                        torch.Generator().manual_seed(seed % (2**63 - 1)).get_state(),
                    )
                    assert policy.random.getstate() == random.Random(seed).getstate()
                else:
                    assert (
                        policy.seed == seed
                        and policy.rng.getstate() == random.Random(seed).getstate()
                    )
            return result

        monkeypatch.setattr(composable, "ProductionSimulator", simulator)
        monkeypatch.setattr(composable, "policies_for", policies)
        import io

        from rich.console import Console

        view = runtime.RuntimeDisplay(
            runtime.DisplayOptions(verbose=False, progress="off"),
            console=Console(file=io.StringIO(), force_terminal=False),
        )
        view.bind(tmp_path / "real-runtime-display")

        def emit_to_display(kind, values, **kwargs):
            progress.append(values)
            view.update("probe", values, **kwargs)

        monkeypatch.setattr(runtime, "emit", emit_to_display)
        rows = composable.evaluate_cases(evaluation, cases)
        assert len(rows) == 1 and rows[0]["status"] == "completed"
        assert rows[0]["case_id"] == "evaluation_001"
        assert rows[0]["seed"] == 404008
        assert actual_worlds == [404001] and actual_roots == [604001]
        expected = dict(
            seed_semantics="data_seed",
            data_seed=404008,
            environment_seed=actual_worlds[0],
            policy_root=actual_roots[0],
        )
        assert all(rows[0][key] == value for key, value in expected.items())
        assert progress and all(
            all(row[key] == value for key, value in expected.items())
            for row in progress
        )
        from smartsom.telemetry.dashboard import facts

        caption = facts(view, view.tasks["probe"])[-1]
        assert "Data 404008 Env 404001 Policy 604001" in caption
        published = json.loads(
            (tmp_path / "real-runtime-display/logs/progress.json").read_text()
        )
        assert all(
            published["tasks"][0]["values"][key] == value
            for key, value in expected.items()
        )
        view.update("probe", dict(stage="sampling", physical_ticks=8))
        assert not set(expected) & view.tasks["probe"]["values"].keys()
        assert (
            cases[0]["scenario"]["seed"] == 404001 and cases[0]["policy_root"] == 604001
        )
    finally:
        session.close()


def _stage_sources(tmp_path):
    directory = tmp_path / "experiments"
    directory.mkdir()
    ids = ["dqn_first", "ppo_after"]
    for stage, (name, algorithm) in enumerate(zip(ids, ("dqn", "ppo"), strict=True)):
        source, document = _source(tmp_path / (algorithm + "-inputs"), algorithm)
        for dependency in ("factory", "workload", "algorithm"):
            document[dependency] = "../" + algorithm + "-inputs/" + dependency + ".yaml"
        document["runtime"]["environment"]["transport"]["matrix"] = (
            "../" + algorithm + "-inputs/travel.yaml"
        )
        document["batch"] = dict(stage=stage, requires=ids[:stage])
        _write(directory / (name + ".yaml"), document)
        source.unlink()
    return directory, ids


def test_real_public_dqn_then_ppo_stage_barrier_and_frozen_resume(tmp_path):
    from smartsom.experiments.tuning_session import verify_identity

    _require_cpu()
    directory, ids = _stage_sources(tmp_path)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    display = ("--log-format", "json", "--progress", "off")
    with _command(
        tmp_path / "batch-cli",
        environment,
        "batch-run",
        directory,
        "--id",
        ids[0],
        "--id",
        ids[1],
        "--no-background",
        *display,
        register_exit_children=True,
    ) as command:
        root = _wait(command, lambda: _allocated(tmp_path / "outputs"))

        def first_stage_running():
            state = _json(root / "batch.json")
            return state if state["stage"] == "stage_0" else None

        active = _wait(command, first_stage_running)
        assert active["files"][ids[1]]["status"] == "queued"
        future = _json(root / active["tune_directories"]["1"] / "batch.json")
        assert not future["segments"]
        assert all(row["status"] == "queued" for row in future["entries"].values())

        def second_stage_running():
            state = _json(root / "batch.json")
            return state if state["stage"] == "stage_1" else None

        active = _wait(command, second_stage_running, timeout=240)
        assert active["files"][ids[0]]["status"] == "completed"
        previous = _json(root / active["tune_directories"]["0"] / "batch.json")
        assert all(row["status"] == "completed" for row in previous["entries"].values())
        _wait(
            command, lambda: command[0].poll() is not None, timeout=240, allow_exit=True
        )
        assert _finish(command)["status"] == "completed"

    _, plan, state = author_batch.load(root)
    assert set(state["tune_directories"]) == {"0", "1"}
    assert plan["selection"]["resolved_ids"] == ids
    assert state["plan_sha256"] == digest(plan)
    immutable = {root / "plan.json": (root / "plan.json").read_bytes()}
    for stage, name in enumerate(ids):
        tune = root / state["tune_directories"][str(stage)]
        tune_state = _json(tune / "batch.json")
        row = tune_state["entries"][name + "__entry-0001"]
        assert row["status"] == "completed" and row["physical_ticks"] == 8
        checkpoint = Path(row["checkpoint"])
        with (checkpoint / "continuation.pkl").open("rb") as stream:
            continuation = pickle.load(stream)
        assert sum(continuation["optimizations"].values()) > 0
        record = _json(checkpoint / "record.json")
        frozen = PreparedComposition(**row["selected_prepared"])
        assert frozen.config.training.algorithm == ("dqn", "ppo")[stage]
        assert (
            verify_identity(frozen, record, checkpoint)["phase"]
            == "experiment_complete"
        )
        assert "initial" in record["tuning"]["controls"]
        immutable.update(
            {
                tune / "batch.json": (tune / "batch.json").read_bytes(),
                checkpoint / "commit.json": (checkpoint / "commit.json").read_bytes(),
                checkpoint / "continuation.pkl": (
                    checkpoint / "continuation.pkl"
                ).read_bytes(),
            }
        )
    for source in directory.glob("*.yaml"):
        source.unlink()
    with _command(
        tmp_path / "resume-cli",
        environment,
        "resume",
        root,
        *display,
        register_exit_children=True,
    ) as command:
        assert _finish(command)["status"] == "completed"
    assert author_batch.load(root)[2]["plan_sha256"] == state["plan_sha256"]
    assert all(path.read_bytes() == data for path, data in immutable.items())
