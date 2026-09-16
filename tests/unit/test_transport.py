from dataclasses import replace

import pytest

from smartsom.domain import (
    TravelTime,
)


@pytest.mark.parametrize(
    "field,value",
    [("ticks", -1), ("ticks", True), ("ticks", 1.5), ("from_node_id", " ")],
)
def test_invalid_matrix_scalars(field, value):
    with pytest.raises(ValueError):
        replace(TravelTime("I", "M1", 1), **{field: value})
