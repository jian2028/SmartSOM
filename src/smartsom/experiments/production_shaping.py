"""Training-only public potentials at committed physical boundaries.

No simulator mutation, future demand, sampled remaining work, latent quality,
policy targets or learner inputs participate in these potentials.
"""

import copy
import math

from smartsom.config.codec import primitive
from smartsom.experiments.passing_reward import PassingRewardLedger, training_contract
from smartsom.experiments.task_reward import shipment_reward

SCHEMA = "smartsom.production-potential-ledger/v1"


def reward_contract(task, smoothing, shaping=None):
    if shaping is None:
        return training_contract(task, smoothing)
    return {
        "schema": "smartsom.shipment-training-reward/v2",
        "objective": primitive(task),
        "passing_rate_smoothing": primitive(smoothing),
        "production_potential_shaping": primitive(shaping),
        "gamma": 1,
        "initial_potential": "subtract each episode's initial public potential",
        "terminal_semantics": "zero potential at true objective episode end",
    }


def make_training_reward(task, smoothing, shaping=None):
    if shaping is not None:
        return ProductionShapingLedger(task, smoothing, shaping)
    return PassingRewardLedger(task, smoothing) if smoothing is not None else None


def _nominal(step):
    # Canonical reference work is independent of chosen mode, realized duration
    # and machine assignment. Both fields are public workload descriptors.
    return float(step.get("reference_ticks") or step["nominal_ticks"])


def progress_potential(view, task, spec, roles):
    current = {}
    released = set(view["released"])
    for job, row in view["jobs"].items():
        demand = row["demand"]
        if demand not in released:
            continue
        # public_view reports the current attempt even on old scrap rows.
        if "attempt" in row and job != f"{demand}/attempt/{row['attempt']}":
            continue
        current[demand] = (job, row)
    shipped = set(view["shipped"])
    total = float(len(shipped & released))
    for demand in sorted(released - shipped):
        if demand not in current:  # Released but still outside physical input.
            continue
        job, row = current[demand]
        if row["quality"] == "FAIL":
            continue
        steps = row["original_steps"]
        completed = row["step"]
        work = sum(_nominal(step) for step in steps[:completed])
        handoffs = completed
        machine = view["machines"].get(row["location"])
        if completed < len(steps):
            if roles.get(row["location"]) == "machine_pre" or (
                machine is not None
                and machine["job"] == job
                and machine["status"] in ("READY", "PROCESSING")
            ):
                handoffs += 1
            if machine is not None and machine["status"] == "PROCESSING":
                # Elapsed active ticks are observable; remaining sampled work
                # and mode-dependent nominal countdowns are deliberately unused.
                work += min(float(machine["elapsed"]), _nominal(steps[completed]))
        denominator = (
            sum(_nominal(step) for step in steps)
            + (len(steps) + 1) * spec.handoff_reference_ticks
        )
        total += min(
            1.0, (work + handoffs * spec.handoff_reference_ticks) / denominator
        )
    return total / task.reference_jobs


def blocking_clocks(view, previous=None):
    previous = previous or {}
    return {
        machine: {
            "job": row["job"],
            "age": previous[machine]["age"] + 1
            if machine in previous and previous[machine]["job"] == row["job"]
            else 0,
        }
        for machine, row in view["machines"].items()
        if row["status"] == "BLOCKED" and row["job"] is not None
    }


def blocking_potential(view, spec, roles, capacities, clocks):
    age_scale = spec.blocking_age_ticks
    machine_pressure = sum(
        min(1.0, row["age"] / age_scale) for row in clocks.values()
    ) / max(1, len(view["machines"]))
    slots = [
        (owner, slot, capacity)
        for owner, entries in capacities.items()
        if roles.get(owner) in ("machine_pre", "machine_post")
        for slot, capacity in entries.items()
        if capacity is not None
    ]
    buffer_pressure = 0.0
    for owner, slot, capacity in slots:
        jobs = view["storage"][owner][slot]
        occupancy = len(jobs) / capacity
        if jobs and occupancy >= spec.buffer_pressure_threshold:
            oldest = max(view["tick"] - view["jobs"][job]["since"] for job in jobs)
            buffer_pressure += occupancy * min(1.0, oldest / age_scale)
    buffer_pressure /= max(1, len(slots))
    cargo_pressure = 0.0
    for vehicle in view["agvs"].values():
        if vehicle["job"] is not None:
            # Job location timestamp changes at physical pickup/drop, never at
            # retargeting. A fixed public grace avoids penalizing normal transit.
            age = view["tick"] - view["jobs"][vehicle["job"]]["since"]
            cargo_pressure += min(1.0, max(0, age - spec.cargo_grace_ticks) / age_scale)
    cargo_pressure /= max(1, len(view["agvs"]))
    return -(machine_pressure + buffer_pressure + cargo_pressure) / 3.0


class ProductionShapingLedger:
    """A unique physical ledger, with exact gamma1 terminal and restore closure."""

    def __init__(self, task, smoothing, shaping):
        self.task, self.smoothing, self.spec = task, smoothing, shaping
        self.base = PassingRewardLedger(task, smoothing) if smoothing else None
        self.contract = reward_contract(task, smoothing, shaping)
        self.episodes = {}

    def _potentials(self, view, roles, capacities, clocks):
        return (
            progress_potential(view, self.task, self.spec, roles)
            if self.spec.progress_weight
            else 0.0,
            blocking_potential(view, self.spec, roles, capacities, clocks)
            if self.spec.blocking_weight
            else 0.0,
        )

    def interval(
        self,
        env,
        episode,
        *,
        tick,
        totals,
        good,
        terminal,
        before,
        after,
        roles,
        capacities,
    ):
        key = (env, episode)
        previous = self.episodes.get(key)
        if previous is None:
            if before["tick"] != 0:
                raise ValueError("production shaping must start at episode tick zero")
            clocks = blocking_clocks(before)
            initial = self._potentials(before, roles, capacities, clocks)
            previous = dict(
                env=env,
                episode=episode,
                tick=0,
                terminal=False,
                initial=list(initial),
                raw=list(initial),
                phi=[0.0, 0.0],
                clocks=clocks,
                progress_return=0.0,
                blocking_return=0.0,
                base_training_return=0.0,
                objective_return=0.0,
                totals={"shipments": 0, "passing_change": 0.0, "overdue_time": 0},
            )
        if (
            previous["terminal"]
            or tick != previous["tick"] + 1
            or before["tick"] != previous["tick"]
            or after["tick"] != tick
        ):
            raise ValueError("inconsistent production shaping physical interval")
        if self.base:
            details = self.base.interval(
                env, episode, tick=tick, totals=totals, good=good, terminal=terminal
            )
            base = details["training_reward"]
            objective = details["objective_reward"]
        else:
            delta = {k: totals[k] - previous["totals"][k] for k in totals}
            base = objective = shipment_reward(self.task, delta)
            details = dict(objective_reward=objective, objective_terminal=terminal)
        clocks = blocking_clocks(after, previous["clocks"])
        raw = self._potentials(after, roles, capacities, clocks)
        effective = [value - start for value, start in zip(raw, previous["initial"])]
        phi = [0.0, 0.0] if terminal else effective
        weights = (self.spec.progress_weight, self.spec.blocking_weight)
        rewards = [w * (p - old) for w, p, old in zip(weights, phi, previous["phi"])]
        self.episodes[key] = {
            **previous,
            "tick": tick,
            "terminal": terminal,
            "raw": list(raw),
            "phi": phi,
            "clocks": clocks,
            "totals": copy.deepcopy(totals),
            "progress_return": previous["progress_return"] + rewards[0],
            "blocking_return": previous["blocking_return"] + rewards[1],
            "base_training_return": previous["base_training_return"] + base,
            "objective_return": previous["objective_return"] + objective,
        }
        return {
            **details,
            "base_training_reward": base,
            "training_reward": base + sum(rewards),
            "production_progress_reward": rewards[0],
            "production_blocking_reward": rewards[1],
            "production_terminal_compensation": -sum(
                w * p for w, p in zip(weights, effective)
            )
            if terminal
            else 0.0,
            "production_progress_potential": raw[0],
            "production_blocking_potential": raw[1],
        }

    def summary(self, env, episode):
        row = copy.deepcopy(self.episodes[(env, episode)])
        base = self.base.summary(env, episode) if self.base else {}
        training = (
            row["base_training_return"]
            + row["progress_return"]
            + row["blocking_return"]
        )
        return {
            **base,
            **row,
            "training_reward_contract": copy.deepcopy(self.contract),
            "training_return": training,
            "production_closure_error": row["progress_return"]
            + row["blocking_return"]
            - self.spec.progress_weight * row["phi"][0]
            - self.spec.blocking_weight * row["phi"][1],
            "pending_production_compensation": -(
                self.spec.progress_weight * row["phi"][0]
                + self.spec.blocking_weight * row["phi"][1]
            ),
        }

    def state_dict(self):
        return {
            "schema": SCHEMA,
            "contract": copy.deepcopy(self.contract),
            "base": self.base.state_dict() if self.base else None,
            "episodes": [
                copy.deepcopy(row) for _, row in sorted(self.episodes.items())
            ],
        }

    def load_state_dict(self, state):
        if (
            type(state) is not dict
            or set(state) != {"schema", "contract", "base", "episodes"}
            or state["schema"] != SCHEMA
            or state["contract"] != self.contract
            or type(state["episodes"]) is not list
        ):
            raise ValueError("production shaping reward contract changed")
        base = PassingRewardLedger(self.task, self.smoothing) if self.base else None
        if base:
            base.load_state_dict(state["base"])
        elif state["base"] is not None:
            raise ValueError("unexpected production base ledger")
        rows = {}
        for row in state["episodes"]:
            key = (row["env"], row["episode"])
            if (
                key in rows
                or any(
                    type(row[k]) is not int or row[k] < 0
                    for k in ("env", "episode", "tick")
                )
                or type(row["terminal"]) is not bool
            ):
                raise ValueError("invalid production shaping ledger identity")
            for name in ("initial", "raw", "phi"):
                if (
                    type(row[name]) is not list
                    or len(row[name]) != 2
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in row[name]
                    )
                ):
                    raise ValueError("invalid production shaping potential")
            expected_phi = (
                [0.0, 0.0]
                if row["terminal"]
                else [v - start for v, start in zip(row["raw"], row["initial"])]
            )
            checks = [(row["phi"][i], expected_phi[i]) for i in range(2)] + [
                (row["progress_return"], self.spec.progress_weight * expected_phi[0]),
                (row["blocking_return"], self.spec.blocking_weight * expected_phi[1]),
                (row["objective_return"], shipment_reward(self.task, row["totals"])),
            ]
            if base:
                base_row = base.episodes.get(key)
                if base_row is None or (base_row["tick"], base_row["terminal"]) != (
                    row["tick"],
                    row["terminal"],
                ):
                    raise ValueError("production shaping base ledger disagrees")
                checks.append(
                    (row["base_training_return"], base_row["training_return"])
                )
            else:
                checks.append((row["base_training_return"], row["objective_return"]))
            if any(
                not math.isfinite(a) or not math.isclose(a, b, abs_tol=1e-9)
                for a, b in checks
            ):
                raise ValueError("production shaping ledger closure is inconsistent")
            if any(
                type(c["age"]) is not int
                or not 0 <= c["age"] <= row["tick"]
                or type(c["job"]) is not str
                for c in row["clocks"].values()
            ):
                raise ValueError("invalid production blocking clock")
            rows[key] = copy.deepcopy(row)
        if base and set(base.episodes) != set(rows):
            raise ValueError("production shaping base ledger identities disagree")
        self.base, self.episodes = base, rows

    def validate_simulators(self, sims, episodes):
        if self.base:
            self.base.validate_simulators(sims, episodes)
        for env, sim in enumerate(sims):
            row = self.episodes.get((env, episodes[env]))
            if row is None and sim.tick == 0:
                continue
            if row is None or (row["tick"], row["terminal"], row["totals"]) != (
                sim.tick,
                sim.done,
                sim.privileged_task_totals(),
            ):
                raise ValueError(
                    "production shaping ledger disagrees with saved episode"
                )
            view = sim.protocol.public_view()
            expected_clocks = {m: c["job"] for m, c in row["clocks"].items()}
            if expected_clocks != {
                m: c["job"] for m, c in blocking_clocks(view).items()
            } or any(
                not math.isclose(a, b, abs_tol=1e-12)
                for a, b in zip(
                    row["raw"],
                    self._potentials(view, sim.roles, sim.capacity, row["clocks"]),
                )
            ):
                raise ValueError(
                    "production shaping potential disagrees with saved state"
                )
