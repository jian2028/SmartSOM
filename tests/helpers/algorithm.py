"""Migrate historical test recipes explicitly; execution always receives v2."""

from smartsom.config.algorithm_migration import migrate_algorithm
from smartsom.config.base_diagnostic import BaseAlgorithm


def algorithm_options(**recipe):
    recipe.setdefault("schema", "smartsom.base-algorithm/v1")
    return BaseAlgorithm.model_validate(migrate_algorithm(recipe))


def with_protocol(options, **fields):
    data = options.model_dump(mode="json", by_alias=True)
    if "allow_stop" in fields:
        value = fields.pop("allow_stop")
        for group in ("pre", "post"):
            data["decision_protocol"][group]["accept"]["allow_stop"] = value
    if "allow_empty_wait" in fields:
        data["decision_protocol"]["dispatcher"]["pickup"]["allow_empty_wait"] = (
            fields.pop("allow_empty_wait")
        )
    if fields:
        raise ValueError(fields)
    return BaseAlgorithm.model_validate(data)
