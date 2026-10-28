"""Bind explicit observation source once; extensions see detached facts only."""

from dataclasses import replace

from smartsom.config.python_module import load_module
from smartsom.domain.public_facts import freeze


def build_observation(options):
    if options.observation is None:
        return None
    module = load_module(options.observation)
    if not callable(getattr(module, "encode", None)):
        raise ValueError("observation module must export encode(facts, decision)")

    def project(facts, decision):
        return replace(decision, inputs=freeze(module.encode(facts, decision)))

    return project
