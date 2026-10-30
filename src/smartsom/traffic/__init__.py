"""Explicit traffic construction, independent of production policy selection."""

from smartsom.config.components import construct


def build_mover(options, routes, seed=0):
    mover = construct(options.traffic.mover, routes=routes, seed=seed)
    return validate_mover(mover)


def validate_mover(mover):
    if any(
        not callable(getattr(mover, n, None))
        for n in ("moves", "get_state", "set_state")
    ):
        raise ValueError("mover requires moves/get_state/set_state")
    return mover
