"""Resource wrapper and recorded execution share grid completion and budget boundaries."""

import json
from dataclasses import replace

import pytest
from test_resource_training import ManualGridDriver, resource_case

from smartsom.experiments.production import execute
from smartsom.learning.episode import EpisodeLimits
from smartsom.trace.production import Recorder, audit


def lifecycle_case(name):
    case, algorithm = resource_case("production_hand")
    wait = False
    if name == "delayed_arrival":
        case = replace(
            case,
            mode="dynamic",
            tick_limit=20,
            demands=tuple(replace(d, release_at=5, reveal_at=5) for d in case.demands),
        )
        limits, reason, tick = EpisodeLimits(), "completed", 20
    elif name == "exact_tick_completed":
        limits, reason, tick = EpisodeLimits(max_ticks=8), "completed", 8
    elif name == "exact_tick_nonterminal":
        limits, reason, tick = EpisodeLimits(max_ticks=7), "budget_exhausted", 7
    elif name == "no_atomic_overshoot":
        limits, reason, tick = EpisodeLimits(max_ticks=6), "budget_exhausted", 6
    elif name == "decision_limit":
        limits, reason, tick = EpisodeLimits(max_decisions=1), "budget_exhausted", 1
    elif name == "policy_wait":
        wait = True
        limits, reason, tick = EpisodeLimits(max_ticks=4), "budget_exhausted", 4
    else:
        buffers = tuple(
            replace(b, storage=replace(b.storage, capacity=0))
            if b.role == "system_input"
            else b
            for b in case.factory.buffers
        )
        case = replace(case, factory=replace(case.factory, buffers=buffers))
        limits, reason, tick = EpisodeLimits(max_ticks=4), "budget_exhausted", 4
    return case, algorithm, limits, wait, reason, tick


@pytest.mark.learning
@pytest.mark.parametrize(
    "name",
    [
        "delayed_arrival",
        "exact_tick_completed",
        "exact_tick_nonterminal",
        "no_atomic_overshoot",
        "decision_limit",
        "policy_wait",
        "blocked_input",
    ],
)
def test_resource_wrapper_and_recorded_driver_share_lifecycle(name, tmp_path):
    pytest.importorskip("ray.rllib")
    from smartsom.learning.production_ray import ResourceProductionEnv

    case, algorithm, limits, wait, reason, tick = lifecycle_case(name)
    driver = ManualGridDriver(case, algorithm, limits=limits, wait=wait)
    env = ResourceProductionEnv(
        {
            "scenario": case,
            "max_jobs": algorithm.max_jobs,
            "gamma": algorithm.gamma,
            "limits": limits,
            "time_scale": algorithm.time_scale,
            "count_scale": algorithm.count_scale,
        }
    )
    env.reset()
    assert env.adapter.sim.tick == 0
    while not driver.env.finished:
        actor, index = driver.env.actor, driver.index()
        if actor == "terminal":
            actor = env._actor()
        before = driver.sim.tick
        driver.env.step(index)
        _, rewards, terminated, truncated, info = env.step({actor: index})
        assert env.adapter.sim.snapshot() == driver.sim.snapshot()
        assert env.adapter.reason == driver.env.reason
        if driver.sim.tick > before:
            assert (
                env.adapter.last_result["actions"] == driver.env.last_result["actions"]
            )
        assert all(isinstance(v, float) for v in rewards.values())
    assert env.adapter.reason == reason and env.adapter.sim.tick == tick
    assert terminated["__all__"] is (reason == "completed")
    assert truncated["__all__"] is (reason != "completed")
    assert all(row["end_reason"] == reason for row in info.values())
    recorded_driver = ManualGridDriver(case, algorithm, limits=limits, wait=wait)
    path = execute(
        case,
        algorithm,
        policy=recorded_driver,
        output_root=tmp_path,
        verbose=False,
        full_replay=True,
    )
    record = json.loads((path / "run.json").read_text())
    assert record["result"] == json.loads(json.dumps(driver.sim.snapshot()))
    assert record["reason"] == reason
    assert record["status"] == ("completed" if reason == "completed" else "truncated")
    checked = audit(path)
    assert checked["learning"]["reason"] == reason
    assert checked["learning"]["decisions"] == driver.env.decisions
    assert checked["status"] == (
        "passed" if reason == "completed" else "partial_verified"
    )
    env.close()
    driver.env.close()
    recorded_driver.env.close()


def test_writer_failure_after_physical_completion_preserves_actual_state_and_cause(
    tmp_path, monkeypatch
):
    case, algorithm = resource_case("production_hand")
    driver = ManualGridDriver(case, algorithm)
    failure = OSError("resource writer failed after physical completion")
    append = Recorder.append
    attempted = []

    def fail(self, row):
        if driver.sim.done:
            attempted.append(row)
            raise failure
        return append(self, row)

    monkeypatch.setattr(Recorder, "append", fail)
    with pytest.raises(OSError) as caught:
        execute(case, algorithm, policy=driver, output_root=tmp_path, verbose=False)
    assert caught.value is failure
    assert len(attempted) == 1 and attempted[0]["state"]["tick"] == 8
    record = json.loads((failure.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["execution_state"]["completed"]
    rows = [
        json.loads(line)
        for line in (failure.run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["tick"] == 7 and not rows[-1]["state"]["completed"]
    assert not (failure.run_dir / "joint_decisions.jsonl").exists()
