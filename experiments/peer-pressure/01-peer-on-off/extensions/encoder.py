"""Same Base encoder and social head for peer ON/OFF; only inputs differ."""

from .observation import project


def create_encoder(*, group, options, pool="mean-max", peer_enabled=True):
    from smartsom.representations.base.encoder import create_encoder as create

    return create(group=group, options=options)


def pack(requests, *, critics, options):
    from smartsom.representations.base.batching import pack as prepare

    return prepare(
        [
            project(r, peer_enabled=options.encoder.params["peer_enabled"])
            for r in requests
        ],
        critics,
        options.count_scale,
        allow_empty_wait=options.allow_empty_wait,
    )


create_encoder.INPUT_SCHEMA = "smartsom.social-observation/v1"
create_encoder.OUTPUT_SCHEMA = "smartsom.candidate-features/v1"
create_encoder.pack = pack
