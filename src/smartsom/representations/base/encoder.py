"""Framework-independent factory declaration for the Base tensor encoder."""


def create_encoder(*, group, options, width=None):
    from .features import BaseEncoder

    return BaseEncoder(options.width if width is None else width)


def pack(requests, *, critics, options):
    from .batching import pack as prepare

    return prepare(
        requests,
        critics,
        options.count_scale,
        allow_empty_wait=options.allow_empty_wait,
    )


create_encoder.INPUT_SCHEMA = "smartsom.base-observation/native-v2"
create_encoder.OUTPUT_SCHEMA = "smartsom.candidate-features/v1"
create_encoder.pack = pack
