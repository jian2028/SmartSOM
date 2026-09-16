"""Independent holding-buffer timelines, capacity ownership and exact replay."""

import pytest

from smartsom.domain.buffers import HoldingBuffer


@pytest.mark.parametrize("capacity", [-1, True, 1.5, float("inf")])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        HoldingBuffer("H", "H", capacity)
