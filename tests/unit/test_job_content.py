"""Model hand examples, optimal couplings and paired frozen workload inputs."""

from fractions import Fraction
from pathlib import Path

import pytest

from smartsom.config.codec import read_model
from smartsom.workloads.job_content import (
    ContentRecipe,
    generate_content,
    job_distance,
    measure_v,
    optimal_transport,
)

ROOT = Path(__file__).resolve().parents[2]


def test_model_edit_distance_hand_examples():
    a = ((1, 3), (2, 2), (3, 6))
    examples = [
        (a, 0),
        (((1, 3), (2, 5), (3, 6)), 0.03),
        (((1, 3), (4, 2), (3, 6)), 0.1),
        (((1, 3), (3, 6), (2, 2)), 0.24),
        (((1, 3), (4, 4), (2, 2), (3, 6)), 0.14),
    ]
    for b, expected in examples:
        assert float(job_distance(a, b, 10)) == pytest.approx(expected)
        assert job_distance(a, b, 10) == job_distance(b, a, 10)


def test_transport_is_optimal_not_greedy_and_preserves_marginals():
    costs = {("a", "x"): 1, ("a", "y"): 2, ("b", "x"): 2, ("b", "y"): 100}
    score, coupling = optimal_transport(
        {"a": Fraction(1, 2), "b": Fraction(1, 2)},
        {"x": Fraction(1, 2), "y": Fraction(1, 2)},
        lambda a, b: costs[a, b],
    )
    assert score == 2 and coupling == {
        ("a", "y"): Fraction(1, 2),
        ("b", "x"): Fraction(1, 2),
    }


def test_transport_matches_independent_lp_oracle():
    scipy = pytest.importorskip("scipy.optimize")
    import random

    rng = random.Random(51)
    for _ in range(20):
        costs = [[rng.randrange(20) for _ in range(4)] for _ in range(3)]
        left = {i: Fraction(1, 3) for i in range(3)}
        right = {j: Fraction(1, 4) for j in range(4)}
        score, _ = optimal_transport(left, right, lambda i, j: costs[i][j])
        constraints = [[int(k // 4 == i) for k in range(12)] for i in range(3)] + [
            [int(k % 4 == j) for k in range(12)] for j in range(4)
        ]
        result = scipy.linprog(
            [v for row in costs for v in row],
            A_eq=constraints,
            b_eq=[1 / 3] * 3 + [1 / 4] * 4,
            bounds=(0, None),
            method="highs",
        )
        assert result.success and float(score) == pytest.approx(result.fun)


def test_history_compares_before_update():
    a = (("a", 3), ("b", 2), ("c", 6))
    b = (("a", 3), ("b", 5), ("c", 6))
    curve = measure_v([{"spec": b}] * 3, {a: Fraction(1)}, 1, 0.5, 10)
    assert list(map(float, curve)) == pytest.approx([0.03, 0.015, 0.0075])


def test_pool_reorder_and_independent_datasets():
    recipe, _ = read_model(
        ROOT / "configs/test/workloads/small_content.yaml", ContentRecipe
    )
    data = generate_content(recipe, 101, "training_000")
    assert data == generate_content(recipe, 101, "training_000")
    assert (
        data["pool_sha256"]
        != generate_content(recipe, 303, "validation_000")["pool_sha256"]
    )
    values = [data["levels"][k]["V"] for k in ("low", "mid", "high")]
    assert 0 < values[0] < values[1] < values[2] < 1 and data["occurrence_bound"] == 60
    original = {j["id"]: j for j in data["pool"]}
    slots = None
    for row in data["levels"].values():
        demands = row["workload"]["demands"]
        assert {d["demand_id"] for d in demands} == set(original)
        arrivals = sorted(d["release_at"] for d in demands)
        assert slots is None or slots == arrivals
        slots = arrivals
        for demand in demands:
            job = original[demand["demand_id"]]
            assert (
                tuple(
                    (s["operation_type"], s["nominal_ticks"]) for s in demand["steps"]
                )
                == job["spec"]
            )
            assert demand["due_at"] - demand["release_at"] == job["allowance_ticks"]
            assert 1 <= len(demand["steps"]) <= 5
            assert all(
                a["operation_type"] != b["operation_type"]
                for a, b in zip(demand["steps"], demand["steps"][1:])
            )
