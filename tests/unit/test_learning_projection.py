"""Current actor observations hide future and latent state, preserving semantic IDs."""

from dataclasses import replace

import pytest
from test_production_runtime import small_scenario

from smartsom.domain.production import Demand, JointCommand, Outage, ProductionStep
from smartsom.engine.production import ProductionSimulator


def test_unrevealed_job_count_attributes_and_outages_do_not_change_public_prefix():
    base = small_scenario(mode="dynamic", tick_limit=10)
    hidden = tuple(
        Demand(
            f"future{i}",
            (ProductionStep("private", "drill", 99 + i),),
            release_at=9,
            reveal_at=8,
            priority=100 + i,
        )
        for i in range(5)
    )
    changed = replace(
        base, demands=base.demands + hidden, outages=(Outage("machine", 8, 10),)
    )
    a, b = ProductionSimulator(base), ProductionSimulator(changed)
    for _ in range(8):
        assert a.decision() == b.decision()
        a.step(JointCommand())
        b.step(JointCommand())
    assert len(b.decision()["announced"]) == 5
    assert not any(key.startswith("future") for key in b.decision()["jobs"])


def test_current_actor_features_do_not_encode_hidden_future_count():
    np = pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from smartsom.learning.production_env import ProductionEnv

    base = small_scenario(mode="dynamic", tick_limit=10)
    future = Demand(
        "future", (ProductionStep("secret", "drill", 999),), release_at=9, reveal_at=9
    )
    a = ProductionEnv(base, 8)
    b = ProductionEnv(replace(base, demands=base.demands + (future,)), 8)
    a.reset()
    b.reset()
    for _ in range(8):
        np.testing.assert_array_equal(a.observation(), b.observation())
        np.testing.assert_array_equal(a.action_masks(), b.action_masks())
        a.step(5)
        b.step(5)


def test_semantic_machine_candidates_do_not_depend_on_collection_order():
    from test_resource_projection import shared_ports

    case = shared_ports()
    changed = replace(
        case,
        demands=tuple(reversed(case.demands)),
        factory=replace(
            case.factory,
            agvs=tuple(reversed(case.factory.agvs)),
            buffers=tuple(reversed(case.factory.buffers)),
            ports=tuple(reversed(case.factory.ports)),
        ),
    )
    a, b = ProductionSimulator(case), ProductionSimulator(changed)
    assert a.decision() == b.decision()
    for command in (
        JointCommand(agvs=(("agv", "INTERACT"),)),
        JointCommand(agvs=(("agv", "RIGHT"),)),
        JointCommand(agvs=(("agv", "INTERACT"),)),
    ):
        a.step(command)
        b.step(command)
    assert a.decision() == b.decision()
    assert a.machine_choices("machine") == [("job0/attempt/1", "normal")]


@pytest.mark.parametrize(
    "bad", [(0, 3, 6), (4, 0, 6), (4, 3, 0), (True, 3, 6), (4, 3, 1.5)]
)
def test_historical_projection_dimensions_remain_strict_read_contracts(bad):
    from smartsom.learning.projection import ProjectionSpec

    with pytest.raises(ValueError):
        ProjectionSpec(*bad)


def test_changed_probability_visibility_is_rejected_before_new_episode():
    pytest.importorskip("gymnasium")
    from smartsom.learning.production_env import ProductionEnv

    case = small_scenario()
    env = ProductionEnv(case, 8)
    env.reset()
    before = env.sim
    env.episode_source = lambda _: replace(
        case, quality_probability_visibility="hidden"
    )
    with pytest.raises(ValueError, match="frozen"):
        env.reset()
    assert env.sim is before
