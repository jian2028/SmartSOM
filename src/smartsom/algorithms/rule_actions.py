"""Resolve named rules to legal semantic candidates before observation encoding."""

from dataclasses import replace
from urllib.parse import quote

from smartsom.domain.base_decisions import BaseCandidate, BaseDecision

PROJECTION_CONTRACT = "smartsom.rule-actions/v4"
RULE_BITS_START = 45
RULE_BITS_WIDTH = 3
_SPECIAL = {("STOP",): "STOP", ("EMPTY_WAIT",): "EMPTY_WAIT"}
_RULES = {
    ("dispatcher", "pickup"): (
        "nearest",
        "most_ready_stock",
        "most_urgent_ready_stock",
        "fewest_commitments",
    ),
    ("dispatcher", "delivery"): (
        "nearest",
        "least_pre_nominal_work",
        "fewest_pre_jobs",
    ),
    ("pre", "accept"): ("job_edd", "nearest", "least_remaining_nominal_work"),
    ("post", "accept"): ("nearest", "longest_continuous_unassigned_idle"),
    ("input", "accept"): ("nearest", "longest_continuous_unassigned_idle"),
    ("pre", "service"): (
        "job_edd",
        "oldest_recorded_arrival",
        "least_remaining_nominal_work",
    ),
    ("post", "service"): (
        "job_edd",
        "longest_job_queue",
        "least_remaining_nominal_work",
        "most_remaining_nominal_work",
        "most_currently_feasible_next_machines",
        "least_next_pre_nominal_work",
    ),
    ("input", "service"): (
        "job_edd",
        "longest_job_queue",
        "least_remaining_nominal_work",
        "most_remaining_nominal_work",
        "most_currently_feasible_next_machines",
        "least_next_pre_nominal_work",
    ),
    ("machine", "process"): (
        "job_edd",
        "shortest_nominal_processing",
        "longest_job_queue",
    ),
}
RULE_CATALOG = dict(_RULES)


def _identity(request, rule, mode):
    mode_key = "none" if mode is None else "value:" + quote(mode, safe="")
    return f"rule:{request.group}:{request.stage}:{rule}:mode={mode_key}"


def _machine_destinations(request, ordinary, values):
    flags = [values[c.identity]["machine_destination"] for c in ordinary]
    if flags and all(flags):
        return True, None
    if not any(flags):
        return False, "nonmachine destinations; machine-only rule excludes Output"
    return False, "mixed machine and nonmachine destinations"


def _score(rule, values):
    fields = {
        "nearest": ("distance", 1),
        "most_ready_stock": ("ready_count", -1),
        "most_urgent_ready_stock": ("ready_min_due", 1),
        "fewest_commitments": ("commitments", 1),
        "fewest_pre_jobs": ("pre_count", 1),
        "least_pre_nominal_work": ("pre_work", 1),
        "longest_continuous_unassigned_idle": ("idle_age", -1),
        "most_remaining_nominal_work": ("remaining_work", -1),
        "most_currently_feasible_next_machines": ("feasible_next_count", -1),
        "least_next_pre_nominal_work": ("next_pre_work", 1),
        "job_edd": ("due_slack", 1),
        "least_remaining_nominal_work": ("remaining_work", 1),
        "longest_job_queue": ("queue_age", -1),
        "shortest_nominal_processing": ("nominal_duration", 1),
        "oldest_recorded_arrival": ("arrival_age", -1),
    }
    key, sign = fields[rule]
    return sign * values[key]


def _validate(request):
    if (request.group, request.stage) not in _RULES:
        raise ValueError(
            f"unsupported Base role/stage: {request.group}/{request.stage}"
        )
    identities = [c.identity for c in request.candidates]
    actions = [c.action for c in request.candidates]
    if len(set(identities)) != len(identities) or len(set(actions)) != len(actions):
        raise ValueError("physical candidate identities/actions must be unique")
    for candidate in request.candidates:
        if candidate.identity.startswith("rule:"):
            raise ValueError("request has already been rule-projected")
        if candidate.action not in _SPECIAL:
            pair = request.group == "machine" or (
                request.group in ("post", "input") and request.stage == "service"
            )
            if len(candidate.action) != (2 if pair else 1):
                raise ValueError("physical action shape does not match role/stage")


def project(
    request: BaseDecision,
    values,
    *,
    rules=None,
    custom=None,
    facts=None,
    merge_aliases=True,
) -> tuple[BaseDecision, dict]:
    """Return legal rule representatives and a detached, JSON-safe audit mapping.

    ``rules`` includes available, aliased and unavailable rules. Only available
    representatives enter the projected candidate list. ``alias_map`` points to a
    representative rule identity. ``selected_candidates`` maps retained identities
    to physical identities/actions; these actions remain directly executable.
    """
    _validate(request)
    catalog = _RULES[request.group, request.stage]
    rules = catalog if rules is None else rules
    custom = custom or {}
    if (
        not rules
        or len(set(rules)) != len(rules)
        or set(rules) - set(catalog) - custom.keys()
    ):
        raise ValueError("invalid rule subset")
    # Builtin codes never change when selecting a subset. Sorted custom names
    # give extension IDs, all included in scientific identity.
    codes = {name: i for i, name in enumerate((*catalog, *sorted(custom)), 1)}
    ordinary = tuple(c for c in request.candidates if c.action not in _SPECIAL)
    audit = {
        "contract": PROJECTION_CONTRACT,
        "group": request.group,
        "stage": request.stage,
        "physical_candidate_count": len(request.candidates),
        "rules": {},
        "selected_candidates": [],
        "alias_map": {},
        "available_rules": [],
        "limitations": [],
    }
    projected = []
    representatives = {}

    def emit(rule, mode, candidate, code, reason=None):
        identity = _identity(request, rule, mode)
        row = {
            "rule": rule,
            "mode": mode,
            "status": "unavailable",
            "representative": None,
            "physical_identity": None,
            "action": None,
            "reason": reason,
        }
        audit["rules"][identity] = row
        if candidate is None:
            if reason and reason not in audit["limitations"]:
                audit["limitations"].append(reason)
            return
        row.update(physical_identity=candidate.identity, action=candidate.action)
        if merge_aliases and candidate.action in representatives:
            row.update(status="alias", representative=representatives[candidate.action])
            audit["alias_map"][identity] = row["representative"]
            return
        representatives[candidate.action] = identity
        row.update(status="available", representative=identity)
        projected.append(
            BaseCandidate(
                identity,
                candidate.action,
                rule_code=code,
            )
        )
        audit["available_rules"].append(identity)
        audit["selected_candidates"].append({"identity": identity, **row})

    modes = (
        sorted({c.action[1] for c in ordinary})
        if request.group == "machine"
        else [None]
    )
    for mode in modes:
        eligible = tuple(c for c in ordinary if mode is None or c.action[1] == mode)
        for rule in rules:
            code = codes[rule]
            if rule in custom:
                chosen = custom[rule](
                    facts,
                    replace(request, inputs=None, rule_audit=None),
                    eligible,
                )
                candidate = next((c for c in eligible if c.identity == chosen), None)
                if chosen is not None and candidate is None:
                    raise ValueError(
                        f"custom rule {rule} returned an illegal candidate: {chosen!r}"
                    )
                emit(
                    rule,
                    mode,
                    candidate,
                    code,
                    "custom rule unavailable" if chosen is None else None,
                )
                continue
            pool = eligible
            reason = None
            if rule == "longest_continuous_unassigned_idle":
                if len({values[c.identity]["idle_age"] for c in eligible}) <= 1:
                    pool = ()
                    reason = "continuous unassigned idle age is constant among legal applicants"
            elif rule in (
                "most_currently_feasible_next_machines",
                "least_next_pre_nominal_work",
            ):
                pool = tuple(
                    c
                    for c in eligible
                    if values[c.identity]["has_feasible"]
                    and not values[c.identity]["terminal"]
                )
                if not pool:
                    reason = "no nonterminal job with a currently feasible next machine"
            elif (
                request.group == "dispatcher"
                and request.stage == "delivery"
                and rule != "nearest"
            ):
                supported, reason = _machine_destinations(request, ordinary, values)
                if not supported:
                    pool = ()
            elif rule == "most_urgent_ready_stock":
                pool = tuple(
                    c for c in eligible if values[c.identity]["ready_count"] > 0
                )
                if not pool:
                    reason = (
                        "no ready stock; zero summary urgency is not a job due date"
                    )
            candidate = (
                min(
                    pool,
                    key=lambda c: (
                        _score(rule, values[c.identity]),
                        -values[c.identity]["arrival_age"]
                        if request.group in ("post", "input")
                        and request.stage == "service"
                        else 0,
                        c.identity,
                        c.action,
                    ),
                )
                if pool
                else None
            )
            emit(
                rule,
                mode,
                candidate,
                code,
                reason
                or ("no legal ordinary candidate" if candidate is None else None),
            )
    for candidate in request.candidates:
        if candidate.action not in _SPECIAL:
            continue
        # Special actions have their own heads and never alias ordinary rules.
        projected.append(candidate)
        row = {
            "rule": _SPECIAL[candidate.action],
            "mode": None,
            "status": "available",
            "representative": candidate.identity,
            "physical_identity": candidate.identity,
            "action": candidate.action,
            "reason": None,
        }
        audit["rules"][candidate.identity] = row
        audit["selected_candidates"].append({"identity": candidate.identity, **row})
    if not projected:
        raise ValueError(
            f"no available rules for {request.group}.{request.stage}; change the explicit subset"
        )
    return replace(
        request,
        rule_audit=audit,
        candidates=tuple(projected),
    ), audit


def resolve(audit_mapping: dict, rule_identity: str) -> tuple[str, ...]:
    """Resolve a retained representative; reject unavailable/aliased choices."""
    if audit_mapping.get("contract") != PROJECTION_CONTRACT:
        raise ValueError("rule projection contract mismatch")
    row = audit_mapping["rules"][rule_identity]
    if row["status"] != "available":
        raise ValueError("rule is not a retained legal representative")
    return tuple(row["action"])


class RuleMetrics:
    """Per-episode sums; rule rates use rule opportunities, not physical ticks."""

    def __init__(self):
        self.stages = {}

    def add(self, request, action, *, selected_identity=None):
        audit = getattr(request, "rule_audit", None)
        if audit is None:
            return
        from collections import Counter

        stage = self.stages.setdefault(request.group + "." + request.stage, {})
        counts = stage.setdefault("counts", Counter())
        counts["decisions"] += 1
        counts["physical_candidates"] += audit["physical_candidate_count"]
        counts["projected_candidates"] += len(request.candidates)
        rows = stage.setdefault("rules", {})
        for identity, item in audit["rules"].items():
            # Identity contains only configured rule, role/stage and processing
            # mode; no job/vehicle names or unbounded episode identities.
            key = item["rule"] + (":" + item["mode"] if item["mode"] else "")
            row = rows.setdefault(key, Counter())
            row["opportunities"] += 1
            row[item["status"]] += 1
            selected = (
                identity == selected_identity
                if selected_identity is not None
                else tuple(item["action"] or ()) == action
                and sum(c.action == action for c in request.candidates) == 1
            )
            if item["status"] == "available" and selected:
                row["selected"] += 1

    def summary(self):
        result = {}
        for point, stage in self.stages.items():
            counts = dict(stage["counts"])
            rules = {}
            for name, row in stage["rules"].items():
                n = row["opportunities"]
                rules[name] = {
                    **dict(row),
                    "availability_rate": (row["available"] + row["alias"]) / n,
                    "alias_rate": row["alias"] / n,
                    "selection_rate": row["selected"] / counts["decisions"],
                }
            result[point] = {**counts, "rules": rules}
        return result
