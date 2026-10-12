"""Private training potential; the original shipment objective remains unchanged."""

import copy
import math

from smartsom.config.codec import primitive
from smartsom.experiments.task_reward import shipment_reward

SCHEMA = "smartsom.passing-rate-ledger/v1"


def training_contract(task, transform):
    return {
        "schema": "smartsom.shipment-training-reward/v1",
        "objective": primitive(task),
        "passing_rate_smoothing": primitive(transform),
        "gamma": 1,
        "terminal_semantics": "objective episode, including task time limit",
    }


class PassingRewardLedger:
    """One ledger per physical environment/episode, independent of role copies."""

    def __init__(self, task, transform):
        self.task, self.transform = task, transform
        self.contract = training_contract(task, transform)
        self.episodes = {}

    def interval(self, env, episode, *, tick, totals, good, terminal):
        key = (env, episode)
        previous = self.episodes.get(key)
        if previous is None:
            previous = dict(
                env=env,
                episode=episode,
                tick=0,
                shipments=0,
                good=0,
                overdue_time=0,
                passing_true=0.0,
                passing_smooth=0.0,
                quality_smoothed_reward=0.0,
                quality_terminal_compensation=0.0,
                objective_return=0.0,
                training_return=0.0,
                terminal=False,
                terminal_compensations=0,
            )
        if previous["terminal"]:
            raise ValueError(
                "objective episode terminal compensation already committed"
            )
        s, tt = totals["shipments"], totals["overdue_time"]
        if (
            type(s) is not int
            or type(good) is not int
            or not 0 <= good <= s
            or s < previous["shipments"]
            or good < previous["good"]
            or tt < previous["overdue_time"]
            or tick != previous["tick"] + 1
        ):
            raise ValueError("inconsistent physical passing reward interval")
        true = good / s if s else 0.0
        if not math.isclose(true, totals["passing_change"], abs_tol=1e-12):
            raise ValueError("oracle passing totals disagree")
        smooth = good / (s + self.transform.pseudocount)
        common = shipment_reward(
            self.task,
            dict(
                shipments=s - previous["shipments"],
                passing_change=0,
                overdue_time=tt - previous["overdue_time"],
            ),
        )
        original = common + self.task.passing_weight * (true - previous["passing_true"])
        dense = self.task.passing_weight * (smooth - previous["passing_smooth"])
        correction = self.task.passing_weight * (true - smooth) if terminal else 0.0
        training = common + dense + correction
        row = {
            **previous,
            "tick": tick,
            "shipments": s,
            "good": good,
            "overdue_time": tt,
            "passing_true": true,
            "passing_smooth": smooth,
            "quality_smoothed_reward": previous["quality_smoothed_reward"] + dense,
            "quality_terminal_compensation": previous["quality_terminal_compensation"]
            + correction,
            "objective_return": previous["objective_return"] + original,
            "training_return": previous["training_return"] + training,
            "terminal": terminal,
            "terminal_compensations": int(terminal),
        }
        self.episodes[key] = row
        return dict(
            training_reward=training,
            objective_reward=original,
            quality_smoothed_reward=dense,
            quality_terminal_compensation=correction,
            passing_true=true,
            passing_smooth=smooth,
            objective_terminal=terminal,
        )

    def summary(self, env, episode):
        row = copy.deepcopy(self.episodes[(env, episode)])
        row["training_reward_contract"] = copy.deepcopy(self.contract)
        row["pending_quality_compensation"] = (
            0.0
            if row["terminal"]
            else self.task.passing_weight
            * (row["passing_true"] - row["passing_smooth"])
        )
        row["quality_closure_error"] = (
            row["quality_smoothed_reward"]
            + row["quality_terminal_compensation"]
            - self.task.passing_weight * row["passing_true"]
        )
        return row

    def state_dict(self):
        return {
            "schema": SCHEMA,
            "contract": copy.deepcopy(self.contract),
            "episodes": [
                copy.deepcopy(row) for _, row in sorted(self.episodes.items())
            ],
        }

    def load_state_dict(self, state):
        if (
            type(state) is not dict
            or set(state) != {"schema", "contract", "episodes"}
            or state.get("schema") != SCHEMA
            or state.get("contract") != self.contract
        ):
            raise ValueError("passing training reward contract changed")
        if type(state["episodes"]) is not list:
            raise ValueError("invalid passing training ledger rows")
        rows = {}
        for row in state.get("episodes", ()):
            key = (row["env"], row["episode"])
            if key in rows or any(
                type(row[k]) is not int or row[k] < 0
                for k in (
                    "env",
                    "episode",
                    "tick",
                    "shipments",
                    "good",
                    "terminal_compensations",
                )
            ):
                raise ValueError("invalid passing training ledger identity")
            if (
                row["good"] > row["shipments"]
                or type(row["terminal"]) is not bool
                or row["terminal_compensations"] != int(row["terminal"])
                or row["overdue_time"] < 0
                or any(
                    not math.isfinite(row[k])
                    for k in (
                        "passing_true",
                        "passing_smooth",
                        "quality_smoothed_reward",
                        "quality_terminal_compensation",
                        "objective_return",
                        "training_return",
                    )
                )
            ):
                raise ValueError("invalid passing training ledger totals")
            true = row["good"] / row["shipments"] if row["shipments"] else 0.0
            smooth = row["good"] / (row["shipments"] + self.transform.pseudocount)
            correction = (
                self.task.passing_weight * (true - smooth) if row["terminal"] else 0.0
            )
            expected = shipment_reward(
                self.task,
                dict(
                    shipments=row["shipments"],
                    passing_change=true,
                    overdue_time=row["overdue_time"],
                ),
            )
            if any(
                not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)
                for left, right in (
                    (row["passing_true"], true),
                    (row["passing_smooth"], smooth),
                    (row["quality_smoothed_reward"], self.task.passing_weight * smooth),
                    (row["quality_terminal_compensation"], correction),
                    (row["objective_return"], expected),
                    (
                        row["training_return"],
                        expected
                        - self.task.passing_weight * (true - smooth)
                        + correction,
                    ),
                )
            ):
                raise ValueError("passing training ledger closure is inconsistent")
            rows[key] = copy.deepcopy(row)
        self.episodes = rows

    def validate_simulators(self, sims, episodes):
        for env, sim in enumerate(sims):
            row = self.episodes.get((env, episodes[env]))
            if row is None and sim.tick == 0:
                continue
            totals = sim.privileged_task_totals()
            if row is None or (
                row["tick"],
                row["shipments"],
                row["good"],
                row["overdue_time"],
                row["terminal"],
            ) != (
                sim.tick,
                totals["shipments"],
                sim.privileged_output_quality()["good_shipped"],
                totals["overdue_time"],
                sim.done,
            ):
                raise ValueError(
                    "passing training ledger disagrees with saved physical episode"
                )
