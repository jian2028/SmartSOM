"""Public observation optimizations preserve validation and boundary ownership."""

import math
from fractions import Fraction

import pytest

from smartsom.learning.extensions import ObservationSpace, _tensor


@pytest.mark.parametrize("value", [0, -7, 2**64, 0.0, -0.0, 1.25, Fraction(1, 3)])
def test_numeric_fast_path_keeps_float_conversion(value):
    actual = _tensor(value)
    assert type(actual) is float
    assert actual.hex() == float(value).hex()


@pytest.mark.parametrize("value", [True, False, "1", None, object()])
def test_numeric_fast_path_does_not_admit_non_real_scalars(value):
    with pytest.raises(ValueError, match="numeric tensor"):
        _tensor(value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_numeric_fast_path_retains_finite_validation(value):
    with pytest.raises(ValueError, match="finite"):
        _tensor(value)


def test_numeric_fast_path_retains_nested_shapes_and_numpy_fallback():
    np = pytest.importorskip("numpy")
    space = ObservationSpace(fields=(("public", (2, 2)),))
    expected = {"public": ((1.0, -0.0), (3.0, 4.0))}
    assert space.validate({"public": np.asarray(expected["public"])}) == expected
    assert _tensor(np.int64(7)) == 7.0
    with pytest.raises(ValueError, match="rectangular"):
        space.validate({"public": [[1], [2, 3]]})
    with pytest.raises(ValueError, match="shape"):
        space.validate({"public": [[1, 2, 3], [4, 5, 6]]})
    with pytest.raises(ValueError, match="keys"):
        space.validate({"private": [[1, 2], [3, 4]]})


def test_public_supply_keeps_frozen_zero_and_refreshes_missing_owner():
    from pathlib import Path

    from smartsom import api
    from smartsom.engine.production import ProductionSimulator

    root = Path(__file__).resolve().parents[2]
    prepared = api.prepare(
        api.load_config(root / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    sim = ProductionSimulator(prepared.scenario, contract="v3")
    protocol = sim.protocol
    owner = protocol.sources[0]
    current = protocol.source_supply(owner)
    protocol.supply = {owner: 0}
    calls = []
    original = protocol.source_supply

    def source_supply(key):
        calls.append(key)
        return original(key)

    protocol.source_supply = source_supply
    first = protocol.public_view()
    assert first["sources"][owner]["supply"] == 0
    assert calls.count(owner) == 1  # the independent live snapshot still queries it
    first["sources"][owner]["ready"].append("detached-only")
    assert "detached-only" not in first["source_supply"][owner]["ready"]
    protocol.supply.clear()
    calls.clear()
    second = protocol.public_view()
    assert second["sources"][owner]["supply"] == current
    assert calls.count(owner) == 1
    # Detached outputs and repeated same-tick reads cannot retain stale supply.
    first["sources"][owner]["supply"] = 99
    assert protocol.public_view()["sources"][owner]["supply"] == current
    snapshot = sim.snapshot

    def without_source_summary(**kwargs):
        view = snapshot(**kwargs)
        del view["source_supply"]
        return view

    sim.snapshot = without_source_summary
    assert protocol.public_view()["sources"] == second["sources"]
