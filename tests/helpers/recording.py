"""A small recording produced by the current simulator and component interface."""

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import generate_scenario
from smartsom.experiments.reservation_runtime import episode
from smartsom.trace.current import CurrentRecording
from tests.unit.test_current_runtime import inputs


def recording(root):
    factory, workload, options = inputs()
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 4})
    scenario = generate_scenario(factory, workload, 71, "normal", 500)
    with CurrentRecording(root) as recorder:
        report, _, _ = episode(
            factory,
            workload,
            options,
            build_policy(options, 7),
            71,
            "normal",
            500,
            scenario=scenario,
            recording=recorder,
            metrics_level="off",
        )
    assert report["completed"] == 4
    return root, scenario
