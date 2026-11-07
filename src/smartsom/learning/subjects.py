"""Learning subjects follow the configured decision mechanism."""

from smartsom.domain.base_decisions import GROUPS


def groups(options):
    return ("central",) if options.control == "centralized" else GROUPS


def model_groups(options):
    """Physical role, template key, and factory role/stage for each network."""
    for group in groups(options):
        if group in options.learning_policy.split_stages:
            stages = (
                ("pickup", "delivery")
                if group == "dispatcher"
                else ("accept", "service")
            )
            for stage in stages:
                yield group, f"{group}__{stage}", f"{group}.{stage}"
        else:
            yield group, group, group


def subject_key(request, split_stages=()):
    """Parameter sharing never joins distinct physical or stage subjects."""
    if request.group in split_stages:
        return request.group, request.owner, request.stage
    return request.group, request.owner
