"""Actual grid observations, hook state, rewards and uncommitted decision prefixes."""

import copy

import pytest
from test_extensions import CountingEncoder, CountingReward
from test_production_runtime import small_scenario

from smartsom.config.codec import digest, primitive
from smartsom.config.extensions import ExtensionRef, ExtensionSpec, RewardSpec
from smartsom.config.production import AlgorithmConfig
from smartsom.learning.extension_evidence import wire_observation
from smartsom.learning.extensions import bind_extensions, register_extension

pytest.importorskip("gymnasium")
from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.learning.episode import EpisodeLimits
from smartsom.learning.production_env import ProductionEnv
from smartsom.learning.production_replay import GridLearningAudit


class SemanticDriver:
    """Unit policy double operating on the same encoded choices as real inference."""

    def __init__(self, scenario, algorithm, limits, *, evidence="hash"):
        self.env = ProductionEnv(
            scenario,
            algorithm.max_jobs,
            provider=algorithm.provider,
            extensions=algorithm.extensions,
            limits=limits,
            time_scale=algorithm.time_scale,
            count_scale=algorithm.count_scale,
            evidence=evidence,
        )
        self.learning_contract = {
            "schema": "smartsom.grid-learning-evidence/v1",
            "algorithm": primitive(algorithm),
            "limits": primitive(limits),
            "observations": evidence,
            "initial_extensions": self.env.hooks.runtime.state_dict()
            if self.env.hooks
            else None,
        }
        self.env.reset()
        self.sim = self.env.sim
        self.policy = GreedyProductionPolicy(scenario.factory, scenario.seed)
        self.target_tick = None

    def step(self):
        env = self.env
        if self.sim.tick != self.target_tick:
            self.rankings = self.policy.rank(self.sim.decision())
            self.command = self.policy.act(self.sim.decision(self.rankings))
            self.target_tick = self.sim.tick
        if env.current is None:
            index = 0
        else:
            role, key = env.actor.split(":", 1)
            if role == "buffer":
                semantic = next(
                    j for j in self.rankings[key] if j in env.rank_remaining[key]
                )
            elif role == "agv":
                semantic = dict(self.command.agvs).get(key, "WAIT")
            elif role == "quality":
                semantic = dict(self.command.quality).get(key, "WAIT")
            else:
                item = dict(self.command.machines).get(key)
                semantic = (item.job_id, item.mode_id) if item and item.job_id else None
            index = env.current[1].index(semantic)
        return env.step(index)

    def next_tick(self):
        before = self.sim.tick
        while self.sim.tick == before and not self.env.finished:
            self.step()
        return self.env.last_result if self.sim.tick > before else None


def extended_driver(resource, stateful, complete, *, evidence="hash", scenario=None):
    provider = "rllib.resource_ppo" if resource else "sb3.maskable_ppo"
    if stateful:
        register_extension(
            "observation",
            "tests.grid-replay-counter",
            "1",
            CountingEncoder,
            stateful=True,
        )
        register_extension(
            "reward", "tests.grid-replay-reward", "1", CountingReward, stateful=True
        )
    extensions = bind_extensions(
        ExtensionSpec(
            observation=ExtensionRef(
                name="tests.grid-replay-counter" if stateful else "builtin.dict",
                version="1",
            ),
            reward=RewardSpec(
                team=ExtensionRef(name="tests.grid-replay-reward", version="1")
                if stateful
                else ExtensionRef(
                    name="builtin.reward_scale",
                    version="1",
                    parameters={"scale": 2.0, "terminal_offset": 5.0},
                ),
                roles={
                    "machine_policy": ExtensionRef(
                        name="builtin.reward_scale",
                        version="1",
                        parameters={"scale": 3.0, "terminal_offset": 7.0},
                    )
                }
                if resource
                else {},
                learner_scale=0.1,
            ),
        ),
        provider,
    )
    algorithm = AlgorithmConfig(provider=provider, max_jobs=8, extensions=extensions)
    driver = SemanticDriver(
        scenario or small_scenario(),
        algorithm,
        EpisodeLimits(max_decisions=100 if complete else 3),
        evidence=evidence,
    )
    return driver, algorithm


def execute(driver):
    rows = []
    while not driver.env.finished:
        row = driver.next_tick()
        if row is not None:
            rows.append(primitive(row))
    return rows, primitive(driver.env.pending_decisions)


def verify(driver, rows, pending):
    auditor = GridLearningAudit(driver.env.scenario, driver.learning_contract)
    for row in rows:
        auditor.append(row)
    auditor.pending(pending)
    assert auditor.env.reason == driver.env.reason
    assert auditor.env.sim.snapshot() == driver.env.sim.snapshot()
    return auditor.result()


@pytest.mark.parametrize("resource", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_extension_policy_actual_input_replay_and_tamper(resource, stateful, complete):
    driver, _ = extended_driver(resource, stateful, complete)
    rows, pending = execute(driver)
    result = verify(driver, rows, pending)
    assert driver.sim.status == "completed" if complete else driver.env.limit_hit
    assert result["decisions"] == driver.env.decisions
    decisions = [d for row in rows for d in row["decisions"]] + pending
    assert decisions and all(d["observations_sha256"] for d in decisions)
    assert any(d["reward"]["research"] != d["reward"]["raw"] for d in decisions)
    assert sum(d["reward"]["raw"] for d in decisions) == driver.env.episode_reward
    assert decisions[-1]["reward"]["reason"] == (
        "completed" if complete else "budget_exhausted"
    )
    if stateful:
        assert decisions[0]["state_before_sha256"] != decisions[0]["state_after_sha256"]
        assert decisions[1]["state_before_sha256"] == decisions[0]["state_after_sha256"]
    for key in (
        "observations_sha256",
        "state_before_sha256",
        "state_after_sha256",
        "actor",
        "mask",
        "candidates",
        "decision_index",
    ):
        changed = copy.deepcopy(rows)
        changed[0]["decisions"][0][key] = "tampered"
        with pytest.raises(ValueError, match="replay mismatch"):
            verify(driver, changed, pending)
    for field in ("raw", "research", "learner", "reason", "physical_dt"):
        changed = copy.deepcopy(rows)
        changed[0]["decisions"][0]["reward"][field] = "tampered"
        with pytest.raises(ValueError, match="replay mismatch"):
            verify(driver, changed, pending)
    changed = copy.deepcopy(rows)
    changed[0]["decisions"].append(changed[0]["decisions"][0])
    with pytest.raises(ValueError, match="coverage|termination"):
        verify(driver, changed, pending)
    changed = copy.deepcopy(rows)
    changed[0]["decisions"] = []
    with pytest.raises(ValueError, match="missing learning"):
        verify(driver, changed, pending)


def test_float32_evidence_hashes_the_wire_value_and_rejects_overflow():
    value = {"public": (0.1, 16777217)}
    assert wire_observation(value) == {"public": (0.10000000149011612, 16777216.0)}
    assert digest(wire_observation(value)) != digest(value)
    for bad in (float("inf"), float("nan"), 1e100):
        with pytest.raises(ValueError, match="float32"):
            wire_observation([bad])


@pytest.mark.parametrize("evidence", ["hash", "full"])
def test_outer_run_auditor_requires_and_checks_extension_evidence(tmp_path, evidence):
    import json

    from smartsom.experiments.production import execute as run
    from smartsom.trace.production import audit, canonical

    driver, algorithm = extended_driver(False, True, True, evidence=evidence)
    directory = run(
        driver.env.scenario,
        algorithm,
        policy=driver,
        output_root=tmp_path,
        verbose=False,
        full_replay=True,
    )
    assert {p.name for p in directory.iterdir()} == {"run.json", "trace.jsonl", "logs"}
    assert audit(directory)["learning"]["decisions"] == driver.env.decisions
    path = directory / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert ("observations" in rows[0]["decisions"][0]) == (evidence == "full")
    rows[0]["decisions"][0]["observations_sha256"] = "0" * 64
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="learning input/reward replay mismatch"):
        audit(directory)
    rows[0].pop("decisions")
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="missing learning"):
        audit(directory)


@pytest.mark.parametrize("resource", [False, True])
def test_truncated_rewards_match_training_environment_and_replay(resource):
    from dataclasses import replace

    from smartsom.domain.factory_design import AGVDesign, Cell

    case = small_scenario()
    case = replace(
        case,
        factory=replace(
            case.factory,
            agvs=(*case.factory.agvs, AGVDesign("other", "Other", Cell(2, 0))),
        ),
    )
    driver, algorithm = extended_driver(resource, True, True, scenario=case)
    driver.env.limits = EpisodeLimits(max_decisions=1)
    driver.learning_contract["limits"] = primitive(driver.env.limits)
    rows, pending = execute(driver)
    assert rows == [] and len(pending) == 1 and driver.sim.tick == 0
    assert pending[0]["reward"]["reason"] == "budget_exhausted"
    assert pending[0]["reward"]["research"] != pending[0]["reward"]["raw"]
    verify(driver, rows, pending)
    training = ProductionEnv(
        case,
        algorithm.max_jobs,
        provider=algorithm.provider,
        extensions=algorithm.extensions,
        limits=driver.env.limits,
        time_scale=algorithm.time_scale,
        count_scale=algorithm.count_scale,
    )
    training.reset()
    _, reward, terminated, truncated, _ = training.step(pending[0]["selected_index"])
    assert not terminated and truncated and reward == pending[0]["reward"]["learner"]
    assert training.hooks.runtime.state_dict() == driver.env.hooks.runtime.state_dict()


@pytest.mark.parametrize("resource", [False, True])
def test_initial_failure_cannot_be_falsely_replayed(resource):
    driver, _ = extended_driver(resource, True, False)
    rows, pending = execute(driver)
    changed = copy.deepcopy(rows)
    changed[0]["decisions"][0]["reward"]["reason"] = "deadlock"
    with pytest.raises(ValueError, match="replay mismatch"):
        verify(driver, changed, pending)


def test_replay_rejects_extra_uncommitted_or_post_terminal_decisions():
    driver, _ = extended_driver(True, True, True)
    rows, pending = execute(driver)
    assert pending == []
    with pytest.raises(ValueError, match="after termination"):
        verify(driver, rows, [rows[-1]["decisions"][-1]])


def test_missing_learning_contract_is_not_silently_a_physics_only_audit(tmp_path):
    import json

    from smartsom.experiments.production import execute as run
    from smartsom.trace.production import audit

    driver, algorithm = extended_driver(False, True, True)
    root = run(
        driver.env.scenario,
        algorithm,
        policy=driver,
        output_root=tmp_path,
        verbose=False,
    )
    path = root / "run.json"
    manifest = json.loads(path.read_text())
    manifest.pop("learning_contract")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing.*learning evidence contract"):
        audit(root)


def test_uncommitted_decisions_remain_in_run_manifest_and_audit(tmp_path):
    import json
    from dataclasses import replace

    from smartsom.domain.factory_design import AGVDesign, Cell
    from smartsom.experiments.production import execute as run
    from smartsom.trace.production import audit

    case = small_scenario()
    case = replace(
        case,
        factory=replace(
            case.factory,
            agvs=(*case.factory.agvs, AGVDesign("other", "Other", Cell(2, 0))),
        ),
    )
    driver, algorithm = extended_driver(True, True, True, scenario=case)
    driver.env.limits = EpisodeLimits(max_decisions=1)
    driver.learning_contract["limits"] = primitive(driver.env.limits)
    root = run(
        case,
        algorithm,
        policy=driver,
        output_root=tmp_path,
        verbose=False,
        full_replay=True,
    )
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "truncated" and manifest["last_tick"] == 0
    assert len(manifest["pending_decisions"]) == 1
    assert (root / "trace.jsonl").read_text() == ""
    verified = audit(root)
    assert (
        verified["status"] == "partial_verified"
        and verified["learning"]["decisions"] == 1
    )
