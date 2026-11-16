"""V2 network contract; Torch is loaded only when constructing or packing."""

from functools import cache

from smartsom.representations.v2.observation import SCHEMA


@cache
def _implementation():
    from smartsom.representations.v2 import _network_torch

    return _network_torch


def pack(requests, *, critics, options):
    return _implementation().pack(requests, critics=critics, options=options)


def pack_arrays(requests, *, critics, options):
    """Build the same numeric fields without creating intermediate Torch tensors."""
    from smartsom.representations.v2._packing import NumpyArrays, pack

    return pack(requests, critics=critics, options=options, arrays=NumpyArrays)


def create_network(*, group, options, value_head, social_enabled=None, social=None):
    return _implementation().create_network(
        group=group,
        options=options,
        value_head=value_head,
        social_enabled=social_enabled,
        social=social,
    )


def __getattr__(name):
    if name in {"Network", "Attention", "Branch", "initialization", "mlp", "_peers"}:
        return getattr(_implementation(), name)
    raise AttributeError(name)


create_network.INPUT_SCHEMA = SCHEMA
create_network.pack = pack
