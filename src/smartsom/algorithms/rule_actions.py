"""Deterministic rule candidates over detached reservation-Base decisions.

Field contract: current public fields in ``engine/base_protocol.py``:
* dispatcher: distance0, quota1, ready_count2, ready_min_due3,
  commitments14, PRE_count15, PRE_nominal_work42; used columns0:45.
* accept/PRE service (vehicle_vector): distance0, recorded_arrival_age3,
  carried_job_due4, carried_remaining_nominal_work5; POST/INPUT accept adds
  continuous unassigned idle age23 (22 stays zero).
* POST/INPUT service pairs: job_due0, remaining_work1, job_queue_age2,
  recorded_vehicle_arrival_age19; current feasible next-machine count22,
  minimum feasible next-PRE nominal work23, has_feasible24, terminal25.
* machine: job_due0, job_queue_age2, rounded machine/mode nominal_duration12;
  used columns0:16. No hidden realized duration is consulted.

Only padding45:48 is replaced, with a stage-local rule code (the original builtin binary codes are preserved). Physical
features, context, sets and exact selected-candidate social links are retained.
Aliases of a physical action are excluded in fixed rule priority order. Tie
breaking uses semantic identities, never candidate positions or random draws.

Arrival age is the public ``arrived_at`` age: acceptance initializes it and
physical port entry refreshes it. Service rules rank legal job/AGV pairs by job
score, then current arrival age, then semantic identity. Idle age is a separate
passive clock; it is not an application/request queue age. STOP and EMPTY_WAIT remain independent legal special actions when enabled. Constant next-processing time is omitted entirely.
"""

import math
from dataclasses import replace
from urllib.parse import quote

from smartsom.domain.base_decisions import FEATURES, BaseCandidate, BaseDecision

PROJECTION_CONTRACT = "smartsom.rule-actions/v3"
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


def _bits(code):
    return (float(code & 1), float((code >> 1) & 1), float(code >> 2))


def _machine_destinations(request, ordinary):
    """Classify homogeneous destination sets without associating row positions.

    make_request emits station_vectors for every legal destination. Machine owners
    emit tags1,2,3; nonmachine owners emit tag0. A mixed set has no detached mapping
    from owner identity to station rows, so machine-only rules are unavailable.
    """
    stations = request.sets[3]
    tags = [row[0] for row in stations]
    if (
        len(tags) == 3 * len(ordinary)
        and ordinary
        and all(tags.count(tag) == len(ordinary) for tag in (1.0, 2.0, 3.0))
    ):
        return True, None
    if len(tags) == len(ordinary) and all(tag == 0 for tag in tags):
        return False, "nonmachine destinations; machine-only rule excludes Output"
    return (
        False,
        "missing/mixed destination tags lack semantic owner-to-station mapping",
    )


def _score(request, rule, candidate):
    f = candidate.features
    if rule == "nearest":
        return f[0]
    if rule == "most_ready_stock":
        return -f[2]
    if rule == "most_urgent_ready_stock":
        return f[3]
    if rule == "fewest_commitments":
        return f[14]
    if rule == "fewest_pre_jobs":
        return f[15]
    if rule == "least_pre_nominal_work":
        return f[42]
    if rule == "longest_continuous_unassigned_idle":
        return -f[23]
    if rule == "most_remaining_nominal_work":
        return -f[1]
    if rule == "most_currently_feasible_next_machines":
        return -f[22]
    if rule == "least_next_pre_nominal_work":
        return f[23]
    if rule == "job_edd":
        return f[4] if request.group == "pre" else f[0]
    if rule == "least_remaining_nominal_work":
        return f[5] if request.group == "pre" else f[1]
    if rule == "longest_job_queue":
        return -f[2]
    if rule == "shortest_nominal_processing":
        return f[12]
    if rule == "oldest_recorded_arrival":
        return -f[3 if request.group == "pre" else 19]
    raise ValueError(f"unsupported rule: {rule}")


def _validate(request):
    if (request.group, request.stage) not in _RULES:
        raise ValueError(
            f"unsupported Base role/stage: {request.group}/{request.stage}"
        )
    if len(request.sets) != 4 or len(request.context) != FEATURES:
        raise ValueError("unsupported Base context/set schema")
    if any(len(row) != FEATURES for rows in request.sets for row in rows):
        raise ValueError("unsupported Base set-row feature width")
    identities = [c.identity for c in request.candidates]
    actions = [c.action for c in request.candidates]
    if len(set(identities)) != len(identities) or len(set(actions)) != len(actions):
        raise ValueError("physical candidate identities/actions must be unique")
    for candidate in request.candidates:
        if candidate.identity.startswith("rule:"):
            raise ValueError("request has already been rule-projected")
        if len(candidate.features) != FEATURES or not all(
            math.isfinite(x) for x in candidate.features
        ):
            raise ValueError("unsupported/nonfinite candidate features")
        if any(candidate.features[RULE_BITS_START:]):
            raise ValueError("candidate padding45:48 is occupied; refuse overwrite")
        if candidate.action not in _SPECIAL:
            pair = request.group == "machine" or (
                request.group in ("post", "input") and request.stage == "service"
            )
            if len(candidate.action) != (2 if pair else 1):
                raise ValueError("physical action shape does not match role/stage")
    if request.social is not None:
        keys = [s.identity for s in request.social]
        if len(keys) != len(set(keys)) or set(keys) != set(identities):
            raise ValueError(
                "social candidate identities must match physical candidates"
            )


def project(
    request: BaseDecision, *, rules=None, custom=None, facts=None
) -> tuple[BaseDecision, dict]:
    """Return legal rule representatives and a detached, JSON-safe audit mapping.

    ``rules`` includes available, aliased and unavailable rules. Only available
    representatives enter the projected candidate list. ``alias_map`` points to a
    representative rule identity. ``selected_candidates`` maps retained identities
    to physical identities/actions; these actions remain directly executable.
    """
    # Projection runs over current protocol fields before custom observation encoding.
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
    physical_social = (
        {s.identity: s for s in request.social} if request.social is not None else None
    )
    audit = {
        "contract": PROJECTION_CONTRACT,
        "group": request.group,
        "stage": request.stage,
        "physical_candidate_count": len(request.candidates),
        "rules": {},
        "selected_candidates": [],
        "alias_map": {},
        "available_rules": [],
        "feature_slots": [45, 46, 47],
        "limitations": [],
    }
    projected = []
    social = []
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
        if candidate.action in representatives:
            row.update(status="alias", representative=representatives[candidate.action])
            audit["alias_map"][identity] = row["representative"]
            return
        representatives[candidate.action] = identity
        row.update(status="available", representative=identity)
        projected.append(
            BaseCandidate(
                identity,
                candidate.action,
                (*candidate.features[:RULE_BITS_START], *_bits(code)),
            )
        )
        audit["available_rules"].append(identity)
        audit["selected_candidates"].append({"identity": identity, **row})
        if physical_social is not None:
            social.append(
                replace(physical_social[candidate.identity], identity=identity)
            )

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
                    replace(request, social=None, inputs=None, rule_audit=None),
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
                if len({c.features[23] for c in eligible}) <= 1:
                    pool = ()
                    reason = "continuous unassigned idle age is constant among legal applicants"
            elif rule in (
                "most_currently_feasible_next_machines",
                "least_next_pre_nominal_work",
            ):
                pool = tuple(
                    c for c in eligible if c.features[24] == 1 and c.features[25] == 0
                )
                if not pool:
                    reason = "no nonterminal job with a currently feasible next machine"
            elif (
                request.group == "dispatcher"
                and request.stage == "delivery"
                and rule != "nearest"
            ):
                supported, reason = _machine_destinations(request, ordinary)
                if not supported:
                    pool = ()
            elif rule == "most_urgent_ready_stock":
                pool = tuple(c for c in eligible if c.features[2] > 0)
                if not pool:
                    reason = (
                        "no ready stock; zero summary urgency is not a job due date"
                    )
            candidate = (
                min(
                    pool,
                    key=lambda c: (
                        _score(request, rule, c),
                        -c.features[19]
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
        if physical_social is not None:
            social.append(physical_social[candidate.identity])
    if not projected:
        raise ValueError(
            f"no available rules for {request.group}.{request.stage}; change the explicit subset"
        )
    return replace(
        request,
        rule_audit=audit,
        candidates=tuple(projected),
        social=tuple(social) if physical_social is not None else None,
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

    def add(self, request, action):
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
            if item["status"] == "available" and tuple(item["action"]) == action:
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
