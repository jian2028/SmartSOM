"""JobShopLab-style sequential yes/no action distribution over legal candidates."""

import math

import pytest

torch = pytest.importorskip("torch")

from smartsom.config.codec import primitive  # noqa: E402
from smartsom.config.extensions import ActorCriticSpec  # noqa: E402
from smartsom.learning.production_models import (  # noqa: E402
    sequential_binary_log_probabilities,
)


def probabilities(scores, mask):
    scores = torch.tensor([scores], dtype=torch.float64)
    mask = torch.tensor([mask])
    log_p = sequential_binary_log_probabilities(scores, mask)
    return torch.where(mask, log_p.exp(), torch.zeros_like(log_p))[0].tolist()


def test_first_yes_matches_hand_computation():
    # p = 0.5, 0.5, then a forced yes on the last legal candidate.
    assert probabilities([0.0, 0.0, 3.0], [True, True, True]) == pytest.approx(
        [0.5, 0.25, 0.25]
    )
    yes = 1 / (1 + math.exp(-1.0))
    assert probabilities([1.0, 1.0], [True, True]) == pytest.approx([yes, 1 - yes])


def test_illegal_candidates_are_skipped_and_mass_sums_to_one():
    p = probabilities([0.0, 5.0, 0.0, -2.0], [True, False, True, True])
    assert p[1] == 0.0
    assert sum(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.5) and p[2] == pytest.approx(0.25)


def test_all_no_selects_the_last_legal_candidate():
    p = probabilities([-50.0, -50.0, -50.0, 9.0], [True, True, True, False])
    assert p == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-12)
    assert probabilities([4.0], [True]) == pytest.approx([1.0])


def test_softmax_default_is_not_serialized():
    assert "action_mode" not in primitive(ActorCriticSpec())
    spec = ActorCriticSpec(action_mode="sequential_binary")
    assert primitive(spec)["action_mode"] == "sequential_binary"


def uniform_probabilities(scores, mask):
    scores = torch.tensor([scores], dtype=torch.float64)
    mask = torch.tensor([mask])
    log_p = sequential_binary_log_probabilities(scores, mask, uniform=True)
    return torch.where(mask, log_p.exp(), torch.zeros_like(log_p))[0].tolist()


@pytest.mark.parametrize(
    "mask",
    [
        [True] * 22,
        [True, False, True, True, False, True],
        [False, True],
        [True],
    ],
)
def test_uniform_prior_starts_from_a_uniform_first_choice(mask):
    p = uniform_probabilities([0.0] * len(mask), mask)
    legal = sum(mask)
    assert p == pytest.approx([1 / legal if m else 0.0 for m in mask])


def test_uniform_prior_keeps_a_normalized_learnable_distribution():
    scores = [2.0, -1.0, 0.5, 3.0, -4.0]
    mask = [True, True, False, True, True]
    p = uniform_probabilities(scores, mask)
    assert sum(p) == pytest.approx(1.0) and p[2] == 0.0
    assert p != pytest.approx(uniform_probabilities([0.0] * 5, mask))
    # The original mode is unchanged.
    assert probabilities([0.0, 0.0, 3.0], [True] * 3) == pytest.approx(
        [0.5, 0.25, 0.25]
    )
