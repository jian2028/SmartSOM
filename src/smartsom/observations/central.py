"""Current public facility state for direct-assignment policies."""


def create_observation():
    return lambda facts, decision: facts.snapshot()


create_observation.OUTPUT_SCHEMA = "smartsom.central-public-facts/v1"
