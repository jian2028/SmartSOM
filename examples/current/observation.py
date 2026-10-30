"""Example custom input: three current public scalars and two candidate scalars."""

INPUT_SCHEMA = "example.public-counts/v1"


def encode(facts, decision):
    return {
        "context": (
            facts["tick"] / 1000,
            len(facts["jobs"]) / 100,
            len(facts["vehicles"]) / 100,
        ),
        "candidates": tuple(tuple(c.features[:2]) for c in decision.candidates),
    }


def create_observation():
    return encode


create_observation.OUTPUT_SCHEMA = INPUT_SCHEMA
