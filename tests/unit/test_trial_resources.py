"""Profile-only static phase requests; no optional frameworks are needed."""

import pytest
from pydantic import ValidationError

from smartsom.config.codec import ConfigurationError
from smartsom.config.launch import resolve_operations
from smartsom.config.trial_resources import TrialResources, check_capacity


def test_rule_panel_reserves_coordinator_and_all_case_workers():
    from smartsom.config.rllib_resources import RLlibResources

    resources = RLlibResources.model_validate(
        {
            "resources": {"num_cpus_for_main_process": 1, "memory_gib": 2},
            "evaluation_config": {"num_cpus_per_env_runner": 0.5, "memory_gib": 1},
        }
    )
    bundles = resources.scientific_task_bundles(evaluation_workers=2)
    assert bundles == [
        {"CPU": 1, "smartsom_memory_gib": 2},
        {"CPU": 0.5, "smartsom_memory_gib": 1},
        {"CPU": 0.5, "smartsom_memory_gib": 1},
    ]
    with pytest.raises(ConfigurationError):
        check_capacity(bundles, {"CPU": 1, "smartsom_memory_gib": 4})
    with pytest.raises(ValueError, match="does not use GPU"):
        RLlibResources.model_validate(
            {"evaluation_config": {"num_gpus_per_env_runner": 1}}
        ).scientific_task_bundles(evaluation_workers=1)


def test_resident_rllib_profile_uses_public_component_arguments():
    from smartsom.config.rllib_resources import RLlibResources

    profile = RLlibResources.model_validate(
        {
            "resources": {"num_cpus_for_main_process": 2, "memory_gib": 3},
            "learners": {"num_learners": 1, "memory_gib": 4},
            "env_runners": {"num_env_runners": 2, "memory_gib": 1},
            "evaluation_config": {"num_cpus_per_env_runner": 0.5, "memory_gib": 2},
        }
    )
    arguments = profile.arguments()
    assert arguments["resources"]["custom_resources_for_main_process"] == {
        "smartsom_memory_gib": 3
    }
    assert arguments["learners"]["custom_resources_per_learner"] == {
        "smartsom_memory_gib": 4
    }
    assert arguments["env_runners"]["custom_resources_per_env_runner"] == {
        "smartsom_memory_gib": 1
    }
    assert profile.evaluation_arguments() == {
        "num_cpus_per_env_runner": 0.5,
        "custom_resources_per_env_runner": {"smartsom_memory_gib": 2},
    }
    assert RLlibResources().evaluation_arguments() == {}


@pytest.mark.parametrize("component", ["learners", "env_runners"])
def test_local_component_memory_is_not_silently_ignored(component):
    from smartsom.config.rllib_resources import RLlibResources

    count = "num_learners" if component == "learners" else "num_env_runners"
    with pytest.raises(ValidationError, match="local"):
        RLlibResources.model_validate({component: {count: 0, "memory_gib": 2}})


def test_workers_are_reserved_in_addition_to_learner():
    bundles = TrialResources().bundles("training", workers=2, threads=1)
    assert bundles == [{"CPU": 1.0}, {"CPU": 1.0}, {"CPU": 1.0}]
    with pytest.raises(ConfigurationError, match="at least 3"):
        TrialResources.model_validate({"training": {"cpus": 2}}).bundles(
            "training", workers=2
        )


def test_memory_request_is_optional_and_aggregate():
    assert TrialResources().bundles("evaluation", workers=2) == [{"CPU": 1.0}] * 3
    configured = TrialResources.model_validate({"evaluation": {"memory_gib": 6.0}})
    bundles = configured.bundles("evaluation", workers=2)
    assert sum(bundle["memory"] for bundle in bundles) == 6 * 1024**3
    check_capacity(bundles, {"CPU": 3, "memory": 6 * 1024**3})
    with pytest.raises(ConfigurationError, match="exceeding"):
        check_capacity(bundles, {"CPU": 2})


def test_stage_requests_are_profile_only(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("trial_resources:\n  training:\n    cpus: 4\n")
    settings, origins = resolve_operations({}, profile=path)
    assert settings["trial_resources"]["training"]["cpus"] == 4
    assert origins["trial_resources.training.cpus"] == str(path)
    with pytest.raises(ConfigurationError, match="profile"):
        resolve_operations({"trial_resources": {"training": {"cpus": 4}}})


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_cpu_requests_are_positive_integer_counts(value):
    with pytest.raises(ValidationError):
        TrialResources.model_validate({"training": {"cpus": value}})


@pytest.mark.parametrize("name", ["GPU", "smartsom_memory_gib"])
def test_missing_cluster_resource_is_zero_capacity(name):
    with pytest.raises(ValueError, match="exceeding total 0"):
        check_capacity([{name: 1}], {"CPU": 8})


def test_fractional_total_cpu_budget_rejected_during_static_profile_validation():
    from smartsom.config.launch import Profile

    with pytest.raises(ValueError, match="whole-number total resources.cpus"):
        Profile.model_validate({"resources": {"cpus": 2.5}})
    assert Profile.model_validate({"resources": {"cpus": 2.0}}).resources.cpus == 2


def test_public_compile_uses_resident_profile_instead_of_old_phase_worker_count(
    tmp_path,
):
    from pathlib import Path

    import yaml

    from smartsom.config.reservation import compile_reservation
    from tests.unit.test_algorithm_components import author

    root = Path(__file__).resolve().parents[2]
    algorithm = yaml.safe_load(
        (root / "configs/algorithms/current/small/base-shared-ppo.yaml").read_text()
    )
    path = author(tmp_path, algorithm)
    document = yaml.safe_load(path.read_text())
    document["concurrency"] = {"evaluation_processes": 20, "evaluation_cpus": 20}
    document["final_evaluation"] = True
    document.setdefault("run", {})["task"] = "train"
    document["drain_frozen"] = True
    path.write_text(yaml.safe_dump(document))
    profile = tmp_path / "profile.yaml"
    values = {
        "resources": {"cpus": 7},
        "rllib": {
            "env_runners": {"num_env_runners": 2},
            "evaluation_num_env_runners": 4,
        },
    }
    profile.write_text(yaml.safe_dump(values))
    compile_reservation(path, profile=profile)
    values["resources"]["cpus"] = 6
    profile.write_text(yaml.safe_dump(values))
    with pytest.raises(ConfigurationError, match="exceeding total 6"):
        compile_reservation(path, profile=profile)


@pytest.mark.parametrize("task", ["smoke", "calibrate"])
def test_scientific_tasks_reserve_driver_without_unused_learning_workers(
    tmp_path, task
):
    from pathlib import Path

    import yaml

    from smartsom.config.reservation import compile_reservation
    from tests.unit.test_algorithm_components import author

    root = Path(__file__).resolve().parents[2]
    algorithm = yaml.safe_load(
        (root / "configs/algorithms/rllib-base-ppo.yaml").read_text()
    )
    path = author(tmp_path, algorithm)
    document = yaml.safe_load(path.read_text())
    document.setdefault("run", {}).update(task=task, methods=["rule", "random"])
    path.write_text(yaml.safe_dump(document))
    profile = tmp_path / "profile.yaml"
    values = {
        "resources": {"cpus": 1},
        "rllib": {"env_runners": {"num_env_runners": 20}},
    }
    profile.write_text(yaml.safe_dump(values))
    compile_reservation(path, profile=profile)
    values["rllib"]["resources"] = {"num_cpus_for_main_process": 2}
    profile.write_text(yaml.safe_dump(values))
    with pytest.raises(ConfigurationError, match="exceeding total 1"):
        compile_reservation(path, profile=profile)
