"""Bounded summaries from an existing policy forward; never used by a policy."""

import copy
import math


def add_metric(state, name, value):
    row = state.setdefault(name, {"sum": 0.0, "count": 0, "unknown": 0})
    if value is None or not math.isfinite(value):
        row["unknown"] += 1
    else:
        row["sum"] += float(value)
        row["count"] += 1


def record_choice(policy, request, scores, distribution, index, exploratory):
    state = policy.decision_diagnostics
    legal = [i for i, candidate in enumerate(request.candidates) if candidate.legal]
    values = scores.detach()[legal]
    ordered = values.sort(descending=True).values
    greedy = int(scores.argmax())
    selected = request.candidates[index]
    vehicle = request.observation.get("agvs", {}).get(request.owner)
    stratum = request.role
    if vehicle is not None:
        stratum += ":loaded" if vehicle.get("job") is not None else ":empty"
        target = vehicle.get("target") or {}
        port_cell = (
            request.observation.get("topology", {})
            .get("ports", {})
            .get(target.get("port"))
        )
        arrived = (
            port_cell is not None
            and tuple(vehicle.get("cell", ())) == tuple(port_cell)
            and not vehicle.get("travel")
        )
        stratum += (
            ":travelling"
            if vehicle.get("travel")
            else ":arrived"
            if arrived
            else ":stationary"
        )
    rows = state.setdefault("strata", {})
    # Fixed role/load/motion vocabulary; no growing target/owner dictionaries.
    metrics = rows.setdefault(stratum, {})
    gap = float(ordered[0] - ordered[1]) if len(legal) > 1 else None
    algorithm = policy.metadata.get("algorithm")
    stochastic_ppo = algorithm == "ppo" and (
        policy.training or not policy.deterministic
    )
    behavior_probability = (
        float(distribution.probs[index])
        if stochastic_ppo
        else policy.epsilon / len(legal) + (1 - policy.epsilon) * int(index == greedy)
        if algorithm == "dqn" and policy.training
        else float(index == greedy)
    )
    for key, value in {
        "legal_choice_count": len(legal),
        "top1_top2_gap": gap,
        "top_tie": float(gap == 0) if gap is not None else None,
        "selected_greedy": int(index == greedy),
        "behavior_probability": behavior_probability,
        "ppo_probability": float(distribution.probs[index])
        if algorithm == "ppo"
        else None,
        "ppo_entropy": float(distribution.entropy()) if algorithm == "ppo" else None,
        "dqn_epsilon": policy.epsilon
        if algorithm == "dqn" and policy.training
        else None,
        "dqn_exploratory_branch": int(exploratory)
        if algorithm == "dqn" and policy.training
        else None,
    }.items():
        add_metric(metrics, key, value)
    examples = state.setdefault("examples", [])
    example = {
        "tick": request.tick,
        "owner": request.owner,
        "role": request.role,
        "phase": request.stage,
        "stratum": stratum,
        "selected": selected.identity,
        "greedy": request.candidates[greedy].identity,
        "top1_top2_gap": gap,
        "legal_choice_count": len(legal),
        "exploratory_branch": exploratory
        if algorithm == "dqn" and policy.training
        else None,
    }
    # First eight non-greedy/tied choices, otherwise first observations. Each
    # identity is capped: diagnostics must not scale with identifier length.
    example = {k: v[:256] if isinstance(v, str) else v for k, v in example.items()}
    if len(examples) < 8:
        examples.append(example)
    state["decisions"] = state.get("decisions", 0) + 1


def summary_state(values, *, history_complete=True):
    state = copy.deepcopy(values)
    state.update(
        schema="smartsom.decision-scores/v1",
        scope="policy group, all observed decisions; fixed role/load/motion strata",
        phase="pre-commit existing policy forward",
        semantics="DQN scores are Q values; categorical-over-Q log_probability is not behavior probability",
        history_complete=history_complete,
        example_limit=8,
    )
    for metrics in state.get("strata", {}).values():
        for row in metrics.values():
            row["mean"] = row["sum"] / row["count"] if row["count"] else None
    return state


def summary(policy):
    return summary_state(
        policy.decision_diagnostics,
        history_complete=getattr(policy, "decision_history_complete", True),
    )
