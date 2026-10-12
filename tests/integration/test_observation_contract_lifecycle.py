"""Eight-tick public-contract lifecycles, with real learners and spawned readers."""

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

from smartsom.config.codec import canonical_json, digest
from smartsom.config.experiment_v3 import PreparedComposition, model_location
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.policies import ModelSelector
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
