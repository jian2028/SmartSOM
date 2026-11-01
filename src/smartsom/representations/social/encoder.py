"""Social candidate features with the existing actor-only pooling contract."""

from smartsom.representations.base.encoder import create_encoder as base_encoder
from smartsom.representations.base.encoder import pack


def create_encoder(*, group, options, pool="attention", width=None):
    if pool not in {"attention", "mean-max"}:
        raise ValueError("unknown social pooling")
    return base_encoder(group=group, options=options, width=width)


create_encoder.INPUT_SCHEMA = "smartsom.social-observation/v1"
create_encoder.OUTPUT_SCHEMA = "smartsom.candidate-features/v1"
create_encoder.pack = pack
