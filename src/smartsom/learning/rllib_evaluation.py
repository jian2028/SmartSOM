"""Fixed scientific panels executed by official resident evaluation EnvRunners."""

from copy import deepcopy
from functools import partial
from statistics import mean


def configure_panel(
    config,
    cases,
    *,
    num_env_runners,
    interval=1,
    split="periodic_evaluation",
    test_cases=None,
    evaluation_resources=None,
):
    """Use the resolved official evaluation policy, without exploration overrides."""
    if type(num_env_runners) is not int or num_env_runners < 1:
        raise ValueError("fixed evaluation requires at least one resident EnvRunner")
    if split not in {"periodic_evaluation", "test"}:
        raise ValueError("evaluation split must be periodic_evaluation or test")
    identities = [case["case_id"] for case in cases]
    if not cases or len(identities) != len(set(identities)):
        raise ValueError("evaluation panel requires nonempty unique case IDs")
    # Official DQN defaults to explore=False; PPO inherits explore=True.
    # Preserve any explicit official configuration instead of inventing epsilon.
    evaluation = config.get_evaluation_config_object()
    mode = "sample" if evaluation.explore else "argmax"
    frozen = deepcopy(cases)
    for case in frozen:
        if case.get("mode", mode) != mode:
            raise ValueError(
                f"evaluation case mode conflicts with resolved official {mode} policy; "
                "migrate the panel instead of adding a second selection panel"
            )
        case["mode"] = mode
    final = None
    if test_cases is not None:
        final = deepcopy(test_cases)
        keys = [case["case_id"] for case in final]
        if not final or len(keys) != len(set(keys)):
            raise ValueError("test panel requires nonempty unique case IDs")
        for case in final:
            if case.get("mode", mode) != mode:
                raise ValueError(
                    "test panel mode conflicts with resolved official evaluation policy"
                )
            case["mode"] = mode
    return config.evaluation(
        evaluation_interval=interval,
        evaluation_num_env_runners=num_env_runners,
        evaluation_parallel_to_training=False,
        evaluation_config={
            **(evaluation_resources or {}),
            "env_config": {
                "smartsom_panel": frozen,
                "smartsom_panel_split": split,
            },
        },
        custom_evaluation_function=FixedPanels(frozen, split, final),
    )


class FixedPanels:
    """Scientific panel selection within RLlib's public evaluation callback.

    The Algorithm's official configuration/checkpoint owns this callable and
    the frozen cases. Switching panels does not rebuild evaluation workers.
    """

    def __init__(self, cases, split, test_cases=None):
        self.cases, self.split, self.test_cases = cases, split, test_cases
        self.final = False

    def select_final(self):
        if self.test_cases is None:
            raise ValueError("no frozen final test panel was configured")
        self.final = True

    def select_periodic(self):
        self.final = False

    def __call__(self, algorithm, env_runner_group):
        return evaluate_panel(
            algorithm,
            env_runner_group,
            cases=self.test_cases if self.final else self.cases,
            split="test" if self.final else self.split,
        )


def summarize_panel(cases, rows):
    """Only the complete fixed panel under its declared policy can select best."""
    identities = [case["case_id"] for case in cases]
    if not cases or len(identities) != len(set(identities)):
        raise ValueError("evaluation panel requires nonempty unique case IDs")
    if sorted(row["case_id"] for row in rows) != sorted(identities):
        raise RuntimeError("fixed evaluation panel returned missing or duplicate cases")
    by_id = {row["case_id"]: row for row in rows}
    rows = [by_id[key] for key in identities]
    if any(row["mode"] != case["mode"] for row, case in zip(rows, cases, strict=True)):
        raise ValueError("evaluation selection mode differs from frozen case")
    modes = {row["mode"] for row in rows}
    if len(modes) != 1 or not modes <= {"sample", "argmax"}:
        raise ValueError(
            "a fixed selection panel must use one declared evaluation policy"
        )
    complete = all(row["joint_cost"] is not None for row in rows)
    metrics = {
        "cases": rows,
        "complete_panel": complete,
        "selection_mode": rows[0]["mode"],
        "joint_cost": mean(row["joint_cost"] for row in rows) if complete else None,
    }
    for name in ("makespan", "mean_tardiness", "final_passing_rate"):
        metrics[name] = mean(row[name] for row in rows) if complete else None
    return metrics


def _evaluate_cases(runner, *, cases):
    import numpy as np
    from ray.rllib.utils.framework import try_import_torch

    from smartsom.learning.rllib_metrics import factory_infos

    torch, _ = try_import_torch()
    rows, env_steps, agent_steps = [], 0, 0
    for case in cases:
        runner.env.unwrapped.call("set_evaluation_case", case)
        # Seed action draws per frozen case; policy selection comes from the
        # resolved official evaluation config, not a custom exploration rule.
        torch.manual_seed(case["action_seed"])
        episodes = runner.sample(num_episodes=1)
        if len(episodes) != 1:
            raise ValueError("fixed evaluation requires one factory per EnvRunner")
        episode = episodes[0]
        final_infos = factory_infos(episode, -1)
        metrics = [
            info["factory_metrics"] for info in final_infos if "factory_metrics" in info
        ]
        if len(metrics) != 1:
            raise ValueError("evaluation episode lacks unique factory metrics")
        # Gym's single-agent vector wrapper represents scalar infos as NumPy
        # scalars. Scientific panel receipts retain the same JSON scalar schema.
        values = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in metrics[0].items()
        }
        rows.append({"case_id": case["case_id"], "mode": case["mode"], **values})
        env_steps += episode.env_steps()
        agent_steps += episode.agent_steps()
    return rows, env_steps, agent_steps


def evaluate_panel(algorithm, env_runner_group, *, cases=None, split=None):
    """Public RLlib custom_evaluation_function: metrics, env steps, agent steps.

    Cases are frozen before the trial starts. No partial-panel average is used
    for checkpoint selection; missing, repeated or truncated cases invalidate
    the complete objective.
    """
    environment = algorithm.config.evaluation_config["env_config"]
    if cases is None:
        cases = environment["smartsom_panel"]
    if split is None:
        split = environment["smartsom_panel_split"]
    identities = [case["case_id"] for case in cases]
    if not cases or len(identities) != len(set(identities)):
        raise ValueError("evaluation panel requires nonempty unique case IDs")
    mode = "sample" if algorithm.evaluation_config.explore else "argmax"
    if any(case["mode"] != mode for case in cases):
        raise ValueError(
            "fixed panel policy differs from resolved official evaluation policy"
        )
    if env_runner_group is None or not env_runner_group.num_remote_env_runners():
        raise ValueError("fixed panels require resident remote evaluation EnvRunners")
    worker_ids = env_runner_group.healthy_env_runner_ids()
    if len(worker_ids) != env_runner_group.num_remote_env_runners():
        raise RuntimeError("evaluation worker unavailable; panel is incomplete")
    results = env_runner_group.foreach_env_runner(
        [
            partial(_evaluate_cases, cases=cases[i :: len(worker_ids)])
            for i in range(len(worker_ids))
        ],
        local_env_runner=False,
        remote_worker_ids=worker_ids,
    )
    rows = [row for batch, _, _ in results for row in batch]
    metrics = summarize_panel(cases, rows)
    metrics["panel_split"] = split
    # New-stack Algorithm.evaluate() publishes from its MetricsLogger, not
    # directly from the callback return value. Item reduction preserves the
    # latest whole panel, including censored/null objectives.
    for name, value in metrics.items():
        algorithm.metrics.log_value(
            ("evaluation", "env_runners", name), value, reduce="item"
        )
    return metrics, sum(item[1] for item in results), sum(item[2] for item in results)
