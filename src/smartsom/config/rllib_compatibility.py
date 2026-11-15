"""Framework-free support checks shared by public check, launch and restore."""

from smartsom.algorithms.base_policy import evaluation_recipe, uses_learning
from smartsom.config.base_diagnostic import BaseAlgorithm, SamplingSettings
from smartsom.config.codec import ConfigurationError, safe_yaml_load
from smartsom.config.trial_resources import TrialResources


def smoke_learner(options, methods):
    learning = False
    for method in methods:
        current = (
            options
            if method in {"ppo", "dqn"} and method == options.method
            else evaluation_recipe(options, method)
        )
        if uses_learning(current):
            if current != options:
                raise ConfigurationError(
                    "smoke learning methods must use the configured RLlib algorithm"
                )
            learning = True
    return learning


def validate_runtime(experiment, options, operations):
    """Reject unsupported execution before Ray import or output allocation.

    General scientific document loading remains usable for historical inspection;
    it is not evidence that a document is executable on the current framework.
    """
    if experiment.run.task == "evaluate" and (
        options.control != "base"
        or not set(experiment.run.methods) <= {"rule", "random"}
    ):
        raise ConfigurationError(
            "standalone evaluation supports named Base rule/random only; learning and central evaluation remain deferred; use training final_evaluation with official best/last checkpoints"
        )
    if experiment.run.checkpoint is not None:
        raise ConfigurationError(
            "run.checkpoint cannot initialize this runtime; native checkpoints are not resumable; use resume RUN for official Tune recovery"
        )
    unsupported = {
        "evaluation.checkpoints": bool(experiment.evaluation.checkpoints),
        "evaluation.window_analysis": experiment.evaluation.window_analysis is not None,
        "evaluation.recovery": experiment.evaluation.recovery.action != "disabled",
        "evaluation.case_seconds": bool(experiment.evaluation.case_seconds),
        "evaluation.common_window_end_tick": experiment.evaluation.common_window_end_tick
        is not None,
        "replay recording": any(
            getattr(experiment.replay, name) != "off"
            for name in ("train", "periodic_evaluation", "evaluation")
        ),
    }
    for field, enabled in unsupported.items():
        if enabled:
            raise ConfigurationError(
                f"{field} is deferred in the RLlib runtime; existing historical artifacts remain readable"
            )
    if (
        TrialResources.model_validate(operations.get("trial_resources", {}))
        != TrialResources()
    ):
        raise ConfigurationError(
            "phase trial_resources were retired; use profile rllib resources for resident components"
        )
    if experiment.sampling != SamplingSettings():
        raise ConfigurationError(
            "native sampling settings were retired; use profile rllib.env_runners"
        )
    learning = experiment.run.task == "train" or (
        experiment.run.task == "smoke"
        and smoke_learner(options, experiment.run.methods)
    )
    if not learning:
        return
    from smartsom.learning.rllib_config import (
        validate_algorithm_training,
        validate_research_adapters,
    )

    for policy in options.production.policies.values():
        if policy.kind != "learning" or policy.learner.target not in {
            "smartsom.learning.rllib_config.ppo_config",
            "smartsom.learning.rllib_config.dqn_config",
        }:
            raise ConfigurationError(
                "migrate learner to smartsom.learning.rllib_config.ppo_config or dqn_config; native checkpoints and learner parameters cannot be resumed by RLlib"
            )
        validate_algorithm_training(
            policy.learner.target.rsplit(".", 1)[-1].removesuffix("_config"),
            policy.learner.params,
        )
        if policy.role_params:
            raise ConfigurationError(
                "native role_params are unsupported; use official algorithm configuration"
            )
    validate_research_adapters(options)


def validate_plan(plan):
    if hasattr(plan, "template"):
        validate_plan(plan.template)
    elif hasattr(plan, "files"):
        for row in plan.files:
            validate_plan(row["compiled"])
    else:
        options = BaseAlgorithm.model_validate(
            safe_yaml_load(dict(plan.documents)["algorithm"])
        )
        validate_runtime(plan.experiment, options, plan.operations)
    return plan
