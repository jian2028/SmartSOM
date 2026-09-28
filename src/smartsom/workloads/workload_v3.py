"""Factory-independent paired content generation and exact finite V construction."""

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from smartsom.config.codec import canonical_json, digest
from smartsom.config.production import WorkloadFile, named_seed
from smartsom.config.workload_v3 import WorkloadV3
from smartsom.domain.production import Demand, ProductionStep
from smartsom.workloads.job_content import edit_units, optimal_transport

GENERATOR_VERSION = "smartsom.paired-workload/v3.1"
CATEGORIES = ("standard", "parameter", "insertion", "mixed")


@dataclass(frozen=True, slots=True)
class MaterializedWorkload:
    workload: WorkloadFile
    provenance: dict
    tick_limit: int
    mode: str


def _units(value, scale):
    return round(Fraction(str(value)) * scale)


def _pool(config, seed, namespace, counts, segments=1):
    scale = 10**config.personalization.decimal_places
    rng = random.Random(seed)
    p = config.personalization
    kinds = sorted({kind for template in config.templates for kind in template.route})
    pool = []
    for segment in range(segments):
        for template in config.templates:
            count = counts.get(template.id, 0)
            categories = [
                category
                for category, weight in zip(CATEGORIES, p.category_weights, strict=True)
                for _ in range(count * weight // sum(p.category_weights))
            ]
            for category in categories:
                times = [_units(t, scale) for t in template.times]
                route = list(template.route)
                original_total = sum(times)
                if category in ("parameter", "mixed"):
                    selected = rng.sample(range(len(route)), rng.randint(1, len(route)))
                    for position in selected:
                        magnitude = rng.uniform(p.adjustment_min, p.adjustment_max)
                        direction = rng.choice((-1, 1))
                        times[position] = max(
                            1,
                            round(
                                times[position]
                                * Fraction(str(1 + direction * magnitude))
                            ),
                        )
                if category in ("insertion", "mixed"):
                    legal = [
                        (gap, kind)
                        for gap in range(len(route) + 1)
                        for kind in kinds
                        if (gap == 0 or route[gap - 1] != kind)
                        and (gap == len(route) or route[gap] != kind)
                    ]
                    if not legal:
                        raise ValueError(
                            f"no legal insertion for template {template.id}"
                        )
                    gap, kind = rng.choice(legal)
                    route.insert(gap, kind)
                    times.insert(
                        gap,
                        max(
                            1,
                            round(
                                original_total
                                * Fraction(
                                    str(rng.uniform(p.insertion_min, p.insertion_max))
                                )
                            ),
                        ),
                    )
                pool.append(
                    {
                        "id": f"{namespace}_job_{len(pool) + 1:06d}",
                        "segment": segment,
                        "base_type": template.id,
                        "category": category,
                        "novel": template.novel,
                        "spec": tuple(zip(route, times, strict=True)),
                    }
                )
    return pool


def _curve(order, history, config, scale, cache):
    history = dict(history)
    eta = Fraction(str(config.volatility.eta))
    bound = math.ceil(config.occurrence_bound * scale)
    size = config.arrivals.window_jobs

    def cost(a, b):
        pair = (a, b) if a <= b else (b, a)
        if pair not in cache:
            cache[pair] = edit_units(a, b, bound)
        return cache[pair]

    values = []
    for start in range(0, len(order), size):
        counts = Counter(job["spec"] for job in order[start : start + size])
        current = {spec: Fraction(count, size) for spec, count in counts.items()}
        value, _ = optimal_transport(current, history, cost)
        values.append(value / (10 * bound))
        history = {
            spec: (1 - eta) * history.get(spec, 0) + eta * current.get(spec, 0)
            for spec in history.keys() | current.keys()
        }
        history = {spec: mass for spec, mass in history.items() if mass}
    return tuple(values)


def _legal_pair(order, rng, window_size):
    """Uniform stable-ID pair sampling with linear auxiliary storage."""
    strata = {}
    for index, job in sorted(enumerate(order), key=lambda row: row[1]["id"]):
        strata.setdefault((job["segment"], job["rush"]), []).append(index)
    groups = []
    total = 0
    for positions in strata.values():
        remaining = Counter(index // window_size for index in positions)
        for offset, first in enumerate(positions):
            window = first // window_size
            remaining[window] -= 1
            count = len(positions) - offset - 1 - remaining[window]
            if count:
                groups.append((total, count, first, positions, offset))
                total += count
    if not total:
        return None
    draw = rng.randrange(total)
    for start, count, first, positions, offset in reversed(groups):
        if start <= draw < start + count:
            target = draw - start
            for second in positions[offset + 1 :]:
                if first // window_size != second // window_size:
                    if target == 0:
                        return first, second
                    target -= 1
    raise AssertionError("legal swap sampler failed")


@lru_cache(maxsize=12)
def _generate(config_json, data_seed, split):
    config = WorkloadV3.model_validate_json(config_json)
    scale = 10**config.personalization.decimal_places
    streams = {
        name: named_seed(data_seed, f"workload-v3/{split}/{name}")
        for name in (
            "content",
            "rush",
            "history",
            "reference-order",
            "low-order",
            "high-order",
        )
    }
    pool = _pool(
        config,
        streams["content"],
        split,
        {t.id: t.count for t in config.templates},
        config.segments,
    )
    for job in pool:
        job["rush"] = (
            random.Random(named_seed(streams["rush"], job["id"])).random()
            < config.due.rush_probability
        )
        normal = (
            Fraction(str(config.due.base_seconds))
            + Fraction(str(config.due.reference_work_factor))
            * sum(t for _, t in job["spec"])
            / scale
            + Fraction(str(config.due.operation_seconds)) * len(job["spec"])
        )
        allowance = normal * (
            Fraction(str(config.due.rush_ratio)) if job["rush"] else 1
        )
        job["normal_allowance_seconds"] = str(normal)
        job["allowance_ticks"] = math.ceil(
            allowance / Fraction(str(config.arrivals.tick_seconds))
        )
    history_counts = config.history.counts or {
        t.id: t.count for t in config.templates if not t.novel
    }
    samples = _pool(config, streams["history"], "initial-history", history_counts)
    hist_counts = Counter(job["spec"] for job in samples)
    history = {
        spec: Fraction(count, len(samples)) for spec, count in hist_counts.items()
    }
    order = []
    reference_rng = random.Random(streams["reference-order"])
    for segment in range(config.segments):
        jobs = [job for job in pool if job["segment"] == segment]
        reference_rng.shuffle(jobs)
        order.extend(jobs)
    cache = {}
    mid_curve = _curve(order, history, config, scale, cache)
    levels = {
        "mid": {
            "order": order,
            "curve": mid_curve,
            "V": sum(mid_curve) / len(mid_curve),
            "accepted": 0,
            "proposals": 0,
            "stop_reason": "reference permutation",
        }
    }
    for name, direction in (("low", -1), ("high", 1)):
        candidate = list(order)
        curve, score = mid_curve, levels["mid"]["V"]
        rng = random.Random(streams[f"{name}-order"])
        accepted = proposals = 0
        stop_reason = "proposal budget exhausted"
        for _ in range(config.volatility.swap_proposals):
            pair = _legal_pair(candidate, rng, config.arrivals.window_jobs)
            if pair is None:
                stop_reason = "no same-class same-segment cross-window pair"
                break
            proposals += 1
            a, b = pair
            candidate[a], candidate[b] = candidate[b], candidate[a]
            new_curve = _curve(candidate, history, config, scale, cache)
            new_score = sum(new_curve) / len(new_curve)
            if direction * (new_score - score) > 0:
                score, curve = new_score, new_curve
                accepted += 1
            else:
                candidate[a], candidate[b] = candidate[b], candidate[a]
        levels[name] = {
            "order": candidate,
            "curve": curve,
            "V": score,
            "accepted": accepted,
            "proposals": proposals,
            "stop_reason": stop_reason,
        }
    separation = Fraction(str(config.volatility.minimum_separation))
    values = [levels[name]["V"] for name in ("low", "mid", "high")]
    if (
        not (values[0] < values[1] < values[2])
        or min(values[1] - values[0], values[2] - values[1]) < separation
    ):
        diagnostics = {
            name: {
                "V": float(row["V"]),
                "accepted": row["accepted"],
                "proposals": row["proposals"],
                "stop_reason": row["stop_reason"],
            }
            for name, row in levels.items()
        }
        raise ValueError(
            "cannot construct separated Low < Mid < High from the frozen pool; "
            f"no Job or rush label was redrawn: {diagnostics}"
        )
    slots = config.arrivals.slots_seconds
    if slots is None:
        slots = tuple(
            0
            if i < config.arrivals.initial_jobs
            else float(i * Fraction(str(config.arrivals.interval_seconds)))
            for i in range(config.total_jobs)
        )
    rush_windows = [
        sum(job["rush"] for job in order[start : start + config.arrivals.window_jobs])
        for start in range(0, len(order), config.arrivals.window_jobs)
    ]
    return {
        "streams": streams,
        "pool": pool,
        "history_samples": samples,
        "initial_history": history,
        "levels": levels,
        "slots": slots,
        "rush_windows": rush_windows,
        "scale": scale,
    }


def materialize_workload(
    config: WorkloadV3, data_seed: int, split: str = "train"
) -> MaterializedWorkload:
    """Compile one frozen V realization; environment/method identities are absent by design."""
    if type(data_seed) is not int or not 0 <= data_seed < 2**64:
        raise ValueError("data_seed must be an unsigned 64-bit integer")
    if not split or not isinstance(split, str):
        raise ValueError("split must be a nonempty declared data namespace")
    selected = config.volatility.level
    normalized = config.model_copy(
        update={"volatility": config.volatility.model_copy(update={"level": "mid"})}
    )
    data = _generate(canonical_json(normalized), data_seed, split)
    row = data["levels"][selected]
    scale, tick_seconds = data["scale"], Fraction(str(config.arrivals.tick_seconds))
    demands = []
    for job, slot in zip(row["order"], data["slots"], strict=True):
        release = math.ceil(Fraction(str(slot)) / tick_seconds)
        reveal = max(
            0,
            release
            - math.ceil(Fraction(str(config.arrivals.notice_seconds)) / tick_seconds),
        )
        demands.append(
            Demand(
                job["id"],
                tuple(
                    ProductionStep(
                        f"op_{index + 1:03d}",
                        kind,
                        math.ceil(Fraction(units, scale) / tick_seconds),
                        reference_ticks=str(Fraction(units, scale) / tick_seconds),
                    )
                    for index, (kind, units) in enumerate(job["spec"])
                ),
                release_at=release,
                due_at=release + job["allowance_ticks"],
                input_id=config.input_id,
                reveal_at=reveal,
                rush=job["rush"],
            )
        )
    levels = {
        name: {
            "V": float(level["V"]),
            "V_exact": str(level["V"]),
            "V_windows": [float(value) for value in level["curve"]],
            "V_windows_exact": list(map(str, level["curve"])),
            "order": [job["id"] for job in level["order"]],
            "accepted_swaps": level["accepted"],
            "proposals": level["proposals"],
            "stop_reason": level["stop_reason"],
        }
        for name, level in data["levels"].items()
    }
    reference_work = Counter()
    for job in data["pool"]:
        for kind, units in job["spec"]:
            reference_work[kind] += units
    provenance = {
        "generator": GENERATOR_VERSION,
        "data_seed": data_seed,
        "split": split,
        "random_streams": data["streams"],
        "config": json.loads(canonical_json(config)),
        "reference_unit_seconds": str(Fraction(1, scale)),
        "pool": data["pool"],
        "pool_sha256": digest(data["pool"]),
        "initial_history_samples": data["history_samples"],
        "initial_history": [
            {"spec": spec, "mass": str(mass)}
            for spec, mass in sorted(data["initial_history"].items())
        ],
        "arrival_slots_seconds": data["slots"],
        "rush_counts_by_window": data["rush_windows"],
        "window_seconds": str(config.arrivals.window_seconds),
        "window_count": config.total_jobs // config.arrivals.window_jobs,
        "category_counts": dict(Counter(job["category"] for job in data["pool"])),
        "rush_count": sum(job["rush"] for job in data["pool"]),
        "realized_rush_fraction": sum(job["rush"] for job in data["pool"])
        / config.total_jobs,
        "reference_work_seconds_by_operation": {
            kind: str(Fraction(units, scale))
            for kind, units in sorted(reference_work.items())
        },
        "occurrence_bound_seconds": str(config.occurrence_bound),
        "edit_cost": "insert/delete=(1+t/Tmax)/2; substitute=(type_mismatch+abs(dt)/Tmax)/2; distance=DP/5",
        "solver": "exact rational min-cost transport; full-series compare-before-update",
        "selected_level": selected,
        "V": levels[selected]["V"],
        "levels": levels,
    }
    # Detach all cached containers so callers cannot corrupt a later paired materialization.
    provenance = json.loads(canonical_json(provenance))
    return MaterializedWorkload(
        WorkloadFile(schema="smartsom.workload/v2", demands=tuple(demands)),
        provenance,
        config.tick_limit or config.default_tick_limit,
        config.mode,
    )
