"""Resource adapter contracts; manual choices are not model-convergence evidence."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_training import bundle as grid_bundle

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.config import resolve_training_run
from smartsom.config.codec import ConfigurationError, primitive
from smartsom.config.experiment import ExperimentConfig, load_config, prepare
from smartsom.config.training import episode_root, framework_seed
from smartsom.experiments.production import execute
from smartsom.learning.checkpoint import file_hash
from smartsom.trace.production import ExecutionAudit, audit

ROOT = Path(__file__).resolve().parents[2]


def test_resource_scientific_sequence_matches_both_central_backends():
    resource = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    for backend in ("sb3", "rllib"):
        central = resolve_training_run(ROOT / f"configs/runs/learning_{backend}.yaml")
        for index in range(5):
            seed = episode_root(101, index)
            assert resource.resolved.episode(seed) == central.resolved.episode(seed)
        assert framework_seed(
            101, resource.resolved.algorithm.provider
        ) != framework_seed(101, central.resolved.algorithm.provider)
    assert json.loads(resource.config_json)["training"]["total_steps"] == 4096


@pytest.mark.parametrize(
    "field", ["action_contract", "observation_contract", "factory_hash"]
)
def test_checkpoint_encoding_checked_before_execution(tmp_path, field):
    prepared, checkpoint = grid_bundle(tmp_path, "marl")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    metadata[field] = "incompatible"
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    authored = json.loads(Path(config.algorithm.source).read_text())
    authored["algorithm"]["checkpoint_sha256"] = file_hash(
        checkpoint / "checkpoint.json"
    )
    Path(config.algorithm.source).write_text(json.dumps(authored))
    with pytest.raises(ConfigurationError, match="incompatible|hash"):
        prepare(config, training=False)
    assert not (tmp_path / "runs").exists()


def test_checkpoint_random_input_compatible_structure_and_corruption_rejected(tmp_path):
    from smartsom.learning.production_contract import validate_model_contract

    prepared, checkpoint = grid_bundle(tmp_path, "marl")
    recipe = prepared.resolved
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    other = recipe.episode(episode_root(101, 3))
    validate_model_contract(metadata, other, recipe.algorithm)
    changed = replace(
        other, factory=replace(other.factory, machines=other.factory.machines[::-1])
    )
    validate_model_contract(metadata, changed, recipe.algorithm)
    with pytest.raises(ValueError, match="incompatible"):
        validate_model_contract(
            metadata, other, recipe.algorithm.model_copy(update={"time_scale": 101})
        )
    (checkpoint / "weights.fixture").write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="digest"):
        prepare(
            ExperimentConfig.model_validate_json(prepared.config_json), training=False
        )
    assert not (tmp_path / "runs").exists()


class ManualGridDriver:
    """Encode declared rule/manual choices through the actual resource decision adapter.

    This tests semantics and evidence without pretending to load model weights.
    Real train/save/load tests live in test_production_workflows.
    """

    def __init__(self, case, algorithm, *, limits=None, wait=False, evidence="full"):
        pytest.importorskip("gymnasium")
        from smartsom.learning.episode import EpisodeLimits
        from smartsom.learning.production_env import ProductionEnv

        self.limits = limits or EpisodeLimits()
        self.env = ProductionEnv(
            case,
            algorithm.max_jobs,
            provider=algorithm.provider,
            limits=self.limits,
            time_scale=algorithm.time_scale,
            count_scale=algorithm.count_scale,
            evidence=evidence,
        )
        self.learning_contract = {
            "schema": "smartsom.grid-learning-evidence/v1",
            "algorithm": primitive(algorithm),
            "limits": primitive(self.limits),
            "observations": evidence,
            "initial_extensions": None,
        }
        self.env.reset()
        self.sim = self.env.sim
        self.rules = GreedyProductionPolicy(
            case.factory, rule="spt", quality_mode=algorithm.quality_mode
        )
        self.wait = wait
        self.commands = None
        self.command_tick = None

    def index(self):
        env = self.env
        if env.current is None:
            return 0
        role, owner = env.actor.split(":", 1)
        if role == "buffer":
            ranked = self.rules.rank(env.sim.decision())[owner]
            selected = next(j for j in ranked if j in env.rank_remaining[owner])
        else:
            if self.command_tick != self.sim.tick:
                self.commands = self.rules.act(env.view)
                self.command_tick = self.sim.tick
            if role == "agv":
                selected = (
                    "WAIT" if self.wait else dict(self.commands.agvs).get(owner, "WAIT")
                )
            elif role == "quality":
                selected = (
                    "WAIT"
                    if self.wait
                    else dict(self.commands.quality).get(owner, "WAIT")
                )
            else:
                command = dict(self.commands.machines).get(owner)
                selected = (
                    (command.job_id, command.mode_id)
                    if command and command.job_id and not self.wait
                    else None
                )
        return env.current[1].index(selected)

    def next_tick(self):
        tick = self.sim.tick
        while self.sim.tick == tick and not self.env.finished:
            self.env.step(self.index())
        return self.env.last_result if self.sim.tick > tick else None


def resource_case(name="quality_production"):
    config = load_config(ROOT / f"configs/runs/{name}.yaml")
    recipe = prepare(config, training=False).resolved
    algorithm = recipe.algorithm.model_copy(
        update={"provider": "rllib.resource_ppo", "max_jobs": 16}
    )
    return recipe.scenario, algorithm


def test_shared_runner_resource_decision_replay_and_tamper_detection(tmp_path):
    case, algorithm = resource_case("production_hand")
    driver = ManualGridDriver(case, algorithm)
    initial = driver.sim.snapshot()
    path = execute(
        case,
        algorithm,
        policy=driver,
        output_root=tmp_path,
        verbose=False,
        full_replay=True,
        observations="full",
    )
    assert audit(path)["status"] == "passed"
    records = [
        json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()
    ]
    assert records[-1]["state"]["tick"] == 8
    assert {p.name for p in path.iterdir()} == {"run.json", "trace.jsonl"}
    for field, value in (
        ("selected_index", 999),
        ("observations_sha256", "0" * 64),
        ("reward", 999),
        ("physical_dt", 99),
    ):
        changed = copy.deepcopy(records)
        changed[0]["decisions"][0][field] = value
        checker = ExecutionAudit(case, initial, driver.learning_contract)
        with pytest.raises(ValueError):
            for row in changed:
                checker.append(row)


def test_waiting_is_explicit_truncation_and_writer_failure_keeps_cause(
    tmp_path, monkeypatch
):
    from smartsom.learning.episode import EpisodeLimits
    from smartsom.trace.production import Recorder

    case, algorithm = resource_case("production_hand")
    limits = EpisodeLimits(max_ticks=3)
    driver = ManualGridDriver(case, algorithm, limits=limits, wait=True)
    path = execute(
        case,
        algorithm,
        policy=driver,
        output_root=tmp_path,
        verbose=False,
        full_replay=True,
    )
    manifest = json.loads((path / "run.json").read_text())
    assert (
        manifest["status"] == "truncated" and manifest["reason"] == "budget_exhausted"
    )
    assert audit(path)["status"] == "partial_verified"
    failure = OSError("resource trace writer failed")
    monkeypatch.setattr(Recorder, "append", lambda *a: (_ for _ in ()).throw(failure))
    driver = ManualGridDriver(case, algorithm)
    with pytest.raises(OSError) as caught:
        execute(case, algorithm, policy=driver, output_root=tmp_path, verbose=False)
    assert caught.value is failure
    record = json.loads((failure.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["execution_state"]["tick"] == 1


@pytest.mark.parametrize(
    "release,outage,processing,capacity",
    [
        (a, b, c, d)
        for a in (False, True)
        for b in (False, True)
        for c in (False, True)
        for d in (False, True)
    ],
)
def test_composed_resource_decisions_replay_one_physical_core(
    release, outage, processing, capacity
):
    from decimal import Decimal

    from smartsom.domain.production import Outage

    case, algorithm = resource_case()
    if release:
        case = replace(
            case,
            mode="dynamic",
            tick_limit=300,
            demands=tuple(
                replace(d, release_at=i * 2, reveal_at=i * 2)
                for i, d in enumerate(case.demands)
            ),
        )
    if outage:
        case = replace(
            case, outages=(Outage(case.factory.machines[0].machine_id, 4, 7),)
        )
    if processing:
        case = replace(
            case, processing_low=Decimal("0.5"), processing_high=Decimal("1.5")
        )
    if capacity:
        buffers = tuple(
            replace(b, storage=replace(b.storage, capacity=1))
            if b.role == "system_input"
            else b
            for b in case.factory.buffers
        )
        case = replace(case, factory=replace(case.factory, buffers=buffers))
    driver = ManualGridDriver(case, algorithm)
    checker = ExecutionAudit(case, driver.sim.snapshot(), driver.learning_contract)
    while not driver.env.finished:
        row = driver.next_tick()
        if row is not None:
            checker.append(row)
    checker.finish_pending(driver.env.pending_decisions)
    assert checker.sim.snapshot() == driver.sim.snapshot()
    assert checker.learning.result()["decisions"] == driver.env.decisions
    assert driver.sim.status == "completed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("modules", ["agv_policy", "machine_policy"]),
        ("modules", ["agv_policy"] * 4),
        ("weights", {"agv_policy": "a" * 64}),
        ("environment_steps", True),
        ("learner_updates", 0),
    ],
)
def test_resource_checkpoint_role_weights_and_counts_are_preflight_errors(
    tmp_path, field, value
):
    prepared, checkpoint = grid_bundle(tmp_path, "marl")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    metadata[field] = value
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    authored = json.loads(Path(config.algorithm.source).read_text())
    authored["algorithm"]["checkpoint_sha256"] = file_hash(
        checkpoint / "checkpoint.json"
    )
    Path(config.algorithm.source).write_text(json.dumps(authored))
    with pytest.raises(ConfigurationError, match="checkpoint"):
        prepare(config, training=False)
    assert not (tmp_path / "runs").exists()
