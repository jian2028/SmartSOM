"""Frozen Job content and measured V; independent of policy and simulator state."""

import heapq
import math
import random
from collections import Counter
from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator

from smartsom.config.codec import digest
from smartsom.config.models import StrictModel
from smartsom.config.production import named_seed


class ContentTemplate(StrictModel):
    id: str
    route: tuple[str, ...]
    times: tuple[int, ...]
    count: int = Field(gt=0)
    novel: bool = False

    @model_validator(mode="after")
    def valid(self):
        if not 1 <= len(self.route) <= 4 or len(self.route) != len(self.times):
            raise ValueError("standard routes need 1-4 aligned operations")
        if any(t < 1 for t in self.times) or self.count % 4:
            raise ValueError("times must be positive; counts divisible by four")
        if any(a == b for a, b in zip(self.route, self.route[1:])):
            raise ValueError("adjacent repeated types are forbidden")
        return self


class ContentRecipe(StrictModel):
    schema_id: Literal["smartsom.job-content/v1"] = Field(alias="schema")
    templates: tuple[ContentTemplate, ...]
    delta: tuple[float, ...] = (-0.2, -0.1, 0.1, 0.2)
    alpha: tuple[float, ...] = (0.1, 0.2, 0.3)
    windows: int = Field(default=16, gt=0)
    jobs_per_window: int = Field(default=4, gt=0)
    window_ticks: int = Field(default=120, gt=0)
    allowance_ticks: int = Field(default=600, gt=0)
    eta: float = Field(default=0.5, gt=0, le=1)
    permutations: int = Field(default=48, ge=3)

    @model_validator(mode="after")
    def valid(self):
        if not self.templates or len({t.id for t in self.templates}) != len(
            self.templates
        ):
            raise ValueError("templates require unique identities")
        if sum(t.count for t in self.templates) != self.windows * self.jobs_per_window:
            raise ValueError("template counts must fill all arrival windows")
        if self.window_ticks < self.jobs_per_window:
            raise ValueError("arrival slots need distinct ticks")
        if (
            not self.delta
            or min(self.delta) <= -1
            or not min(self.delta) < 0 < max(self.delta)
        ):
            raise ValueError(
                "delta must contain both negative and positive adjustments"
            )
        if not self.alpha or min(self.alpha) <= 0:
            raise ValueError("alpha must be positive")
        if not any(not t.novel for t in self.templates):
            raise ValueError("declare common templates for independent initial history")
        return self

    @property
    def occurrence_bound(self):
        return max(
            max(
                math.ceil(t * (1 + Fraction(str(max(self.delta)))))
                for p in self.templates
                for t in p.times
            ),
            max(
                math.ceil(Fraction(str(max(self.alpha))) * sum(p.times))
                for p in self.templates
            ),
        )


def edit_units(a, b, bound):
    """DP cost in integer units of 1/(2*Tmax); d is this cost/(10*Tmax)."""
    old = [0]
    for _, t in b:
        old.append(old[-1] + bound + t)
    for kind, time in a:
        row = [old[0] + bound + time]
        for i, (other, other_time) in enumerate(b, 1):
            row.append(
                min(
                    old[i] + bound + time,
                    row[-1] + bound + other_time,
                    old[i - 1] + bound * (kind != other) + abs(time - other_time),
                )
            )
        old = row
    return old[-1]


def job_distance(a, b, bound):
    return Fraction(edit_units(a, b, bound), 10 * bound)


def optimal_transport(left, right, cost):
    """Exact min-cost flow with rational masses and integer costs, no solver extra.

    Successive shortest paths use residual reverse edges and reduced-cost
    potentials. Capacities are integers after a common denominator conversion.
    Return both objective and coupling, so marginals can be independently checked.
    """
    left_keys, right_keys = sorted(left), sorted(right)
    denominator = math.lcm(*(v.denominator for v in (*left.values(), *right.values())))
    if sum(left.values()) != 1 or sum(right.values()) != 1:
        raise ValueError("transport distributions must each sum to one")
    n = 2 + len(left_keys) + len(right_keys)
    sink = n - 1
    graph = [[] for _ in range(n)]

    def edge(a, b, capacity, price):
        f = [b, len(graph[b]), capacity, price]
        rev = [a, len(graph[a]), 0, -price]
        graph[a].append(f)
        graph[b].append(rev)
        return f

    for i, key in enumerate(left_keys, 1):
        edge(0, i, int(left[key] * denominator), 0)
    coupling_edges = {}
    for i, a in enumerate(left_keys, 1):
        for j, b in enumerate(right_keys, 1 + len(left_keys)):
            coupling_edges[(a, b)] = edge(i, j, denominator, cost(a, b))
    for j, key in enumerate(right_keys, 1 + len(left_keys)):
        edge(j, sink, int(right[key] * denominator), 0)
    potential = [0] * n
    remaining = denominator
    objective = 0
    while remaining:
        distances = [math.inf] * n
        distances[0] = 0
        previous = [None] * n
        queue = [(0, 0)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for index, e in enumerate(graph[node]):
                target, _, capacity, price = e
                candidate = distance + price + potential[node] - potential[target]
                if capacity and candidate < distances[target]:
                    distances[target] = candidate
                    previous[target] = (node, index)
                    heapq.heappush(queue, (candidate, target))
        if previous[sink] is None:
            raise ValueError("no complete transport coupling")
        for i, d in enumerate(distances):
            if d != math.inf:
                potential[i] += d
        flow = remaining
        node = sink
        while node:
            parent, index = previous[node]
            flow = min(flow, graph[parent][index][2])
            node = parent
        node = sink
        while node:
            parent, index = previous[node]
            e = graph[parent][index]
            objective += flow * e[3]
            e[2] -= flow
            graph[node][e[1]][2] += flow
            node = parent
        remaining -= flow
    coupling = {
        pair: Fraction(denominator - e[2], denominator)
        for pair, e in coupling_edges.items()
        if e[2] < denominator
    }
    if any(
        sum(v for (a, _), v in coupling.items() if a == key) != left[key]
        for key in left_keys
    ) or any(
        sum(v for (_, b), v in coupling.items() if b == key) != right[key]
        for key in right_keys
    ):
        raise AssertionError("transport marginal mismatch")
    return Fraction(objective, denominator), coupling


def measure_v(order, history, jobs_per_window, eta, bound):
    history = dict(history)
    rate = Fraction(str(eta))
    curve = []
    cache = {}

    def cost(a, b):
        pair = (a, b)
        if pair not in cache:
            cache[pair] = edit_units(a, b, bound)
        return cache[pair]

    for start in range(0, len(order), jobs_per_window):
        counts = Counter(
            tuple(map(tuple, j["spec"])) for j in order[start : start + jobs_per_window]
        )
        current = {k: Fraction(v, jobs_per_window) for k, v in counts.items()}
        value, _ = optimal_transport(current, history, cost)
        curve.append(value / (10 * bound))
        history = {
            k: (1 - rate) * history.get(k, 0) + rate * current.get(k, 0)
            for k in history.keys() | current.keys()
        }
        history = {k: v for k, v in history.items() if v}
    return tuple(curve)


def generate_content(recipe, seed, label):
    rng = random.Random(named_seed(seed, "content-pool:" + label))
    kinds = sorted({k for t in recipe.templates for k in t.route})
    pool = []
    for template in recipe.templates:
        for i in range(template.count):
            category = ("standard", "parameter", "insertion", "mixed")[i % 4]
            times = list(template.times)
            route = list(template.route)
            if category in ("parameter", "mixed"):
                times = [
                    max(
                        1,
                        round(
                            Fraction(t) * (1 + Fraction(str(rng.choice(recipe.delta))))
                        ),
                    )
                    for t in times
                ]
            if category in ("insertion", "mixed"):
                legal = [
                    (gap, k)
                    for gap in range(len(route) + 1)
                    for k in kinds
                    if (gap == 0 or route[gap - 1] != k)
                    and (gap == len(route) or route[gap] != k)
                ]
                gap, kind = rng.choice(legal)
                route.insert(gap, kind)
                times.insert(
                    gap,
                    max(
                        1,
                        round(
                            Fraction(sum(template.times))
                            * Fraction(str(rng.choice(recipe.alpha)))
                        ),
                    ),
                )
            pool.append(
                {
                    "id": f"{label}_job_{len(pool) + 1:03d}",
                    "base_type": template.id,
                    "category": category,
                    "novel": template.novel,
                    "spec": tuple(zip(route, times)),
                    "allowance_ticks": recipe.allowance_ticks,
                }
            )
    common = [tuple(zip(t.route, t.times)) for t in recipe.templates if not t.novel]
    initial = Counter(common)
    history = {k: Fraction(v, len(common)) for k, v in initial.items()}
    clustered = sorted(pool, key=lambda j: (j["spec"], j["id"]))
    candidates = [clustered, list(reversed(clustered))]
    for index in range(recipe.permutations - 2):
        order = list(pool)
        random.Random(named_seed(seed, f"content-order:{label}:{index}")).shuffle(order)
        # Include coarse clustered and interleaved proposals, then measure every proposal.
        if index < 12:
            blocks = [
                clustered[x : x + recipe.jobs_per_window]
                for x in range(0, len(pool), recipe.jobs_per_window)
            ]
            random.Random(named_seed(seed, f"content-block:{label}:{index}")).shuffle(
                blocks
            )
            order = [job for block in blocks for job in block]
        candidates.append(order)
    scored = []
    for index, order in enumerate(candidates):
        curve = measure_v(
            order, history, recipe.jobs_per_window, recipe.eta, recipe.occurrence_bound
        )
        scored.append((sum(curve) / len(curve), index, curve, order))
    unique = {}
    for row in sorted(scored):
        unique.setdefault(row[0], row)
    ranked = list(unique.values())
    if len(ranked) < 3:
        raise ValueError("cannot construct strictly ordered Low/Mid/High V")
    levels = {}
    for name, row in zip(
        ("low", "mid", "high"),
        (ranked[0], ranked[len(ranked) // 2], ranked[-1]),
        strict=True,
    ):
        value, index, curve, order = row
        demands = []
        for position, job in enumerate(order):
            release = (position // recipe.jobs_per_window) * recipe.window_ticks + (
                position % recipe.jobs_per_window
            ) * recipe.window_ticks // recipe.jobs_per_window
            demands.append(
                {
                    "demand_id": job["id"],
                    "steps": [
                        {
                            "operation_id": f"op_{i + 1:03d}",
                            "operation_type": kind,
                            "nominal_ticks": ticks,
                        }
                        for i, (kind, ticks) in enumerate(job["spec"])
                    ],
                    "release_at": release,
                    "reveal_at": release,
                    "due_at": release + job["allowance_ticks"],
                    "priority": 1,
                }
            )
        levels[name] = {
            "workload": {"schema": "smartsom.workload/v2", "demands": demands},
            "V": float(value),
            "V_windows": [float(v) for v in curve],
            "permutation": index,
        }
    totals = Counter()
    for job in pool:
        for kind, ticks in job["spec"]:
            totals[kind] += ticks
    return {
        "pool": pool,
        "pool_sha256": digest(pool),
        "seed": seed,
        "initial_history": [{"spec": k, "mass": str(v)} for k, v in history.items()],
        "occurrence_bound": recipe.occurrence_bound,
        "reference_work": dict(totals),
        "levels": levels,
    }
