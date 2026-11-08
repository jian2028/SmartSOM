"""Author matched V2 panels and seed-ordered official batch recipes."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import yaml

from smartsom.config.base_diagnostic import generate_scenario
from smartsom.config.codec import digest, primitive
from smartsom.config.scenario_authoring import ScenarioFile

ROOT = Path(__file__).resolve().parents[1]
DRAIN_TICKS = 10000
METHODS = {
    "base": ("v2-base-ddqn", 3),
    "isl": ("v2-isl-ddqn", 3),
    "morris": ("v2-morris-e1a-current-ppo", 3),
    "justin": ("justin", 3),
    "random": ("v2-base-random", 3),
}


def extended_seed(scale, condition, index, purpose):
    key = f"smartsom-v2-panel/v1/{scale}/{condition}/{index}/{purpose}"
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


def build():
    for arm in ("base", "isl"):
        source = ROOT / f"configs/algorithms/v2-{arm}-ddqn.yaml"
        algorithm = yaml.safe_load(source.read_text())
        algorithm["production"]["policies"]["dqn"]["parameter_sharing"] = {
            "default": "independent",
            "roles": {},
        }
        target = source.with_name(f"v2-{arm}-independent-ddqn.yaml")
        target.write_text(yaml.safe_dump(algorithm, sort_keys=False))
    random = yaml.safe_load((ROOT / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    random["production"] = {
        "default": "random",
        "bindings": {},
        "policies": {
            "random": {
                "kind": "policy",
                "observation": {
                    "target": "smartsom.observations.rules.create_random_observation"
                },
                "implementation": {"target": "smartsom.policies.random.create_policy"},
            }
        },
    }
    (ROOT / "configs/algorithms/v2-base-random.yaml").write_text(
        yaml.safe_dump(random, sort_keys=False)
    )
    for scale in ("small", "medium", "large"):
        prior_scale = scale if scale != "large" else "medium"
        prior = ROOT / f"experiments/implicit-sl/{prior_scale}/base/experiment.yaml"
        template = yaml.safe_load(prior.read_text())
        scene = ScenarioFile.model_validate(
            yaml.safe_load((ROOT / f"configs/scenarios/v2/{scale}.yaml").read_text())
        )
        envelope, workload = scene.expand()
        cases = []
        for condition in ("normal", "low", "medium", "high"):
            old = [c for c in template["test"]["cases"] if c["condition"] == condition]
            for index in range(20):
                retained = scale != "large" and index < len(old)
                seed = (
                    old[index]["seed"]
                    if retained
                    else extended_seed(scale, condition, index, "world")
                )
                actions = (
                    old[index]["action_seeds"]
                    if retained
                    else [extended_seed(scale, condition, index, "action")]
                )
                world = generate_scenario(
                    envelope.factory, workload, seed, condition, DRAIN_TICKS
                )
                cases.append(
                    {
                        "id": f"test-v2-{condition}-{index:02d}",
                        "condition": condition,
                        "seed": seed,
                        "action_seeds": actions,
                        "scenario_sha256": digest(primitive(world)),
                    }
                )
        template["drain_ticks"] = DRAIN_TICKS
        template["test"] = {
            "conditions": ["normal", "low", "medium", "high"],
            "cases_per_condition": 20,
            "cases": cases,
        }
        template["scenario"] = f"../../../../configs/scenarios/v2/{scale}.yaml"
        template["tracking"]["group"] = f"v2-{scale}"
        for stage, seed in enumerate((101, 102, 103), 1):
            for method, (algorithm, workers) in METHODS.items():
                if method in ("justin", "random") and stage != 1:
                    continue
                recipe = deepcopy(template)
                recipe["algorithm"] = f"../../../../configs/algorithms/{algorithm}.yaml"
                recipe["train_seeds"] = [seed]
                recipe["batch"] = {"stage": stage, "parallel_files": 3}
                recipe["concurrency"] = {
                    "mode": "manual",
                    "max_concurrent": 3,
                    "evaluation_processes": workers,
                }
                if method not in ("justin", "random"):
                    recipe["checkpoint_retention"] = "best_last"
                    recipe["checkpoint_every_episodes"] = 50
                    recipe["validation_cases"] = 12
                    recipe["validation"] = {
                        "unit": "episodes",
                        "milestones": list(range(50, 1001, 50)),
                        "modes": ["argmax"],
                        "sample_action_seeds": 1,
                        "selection": "argmax_complete_cost",
                        "select_initial": False,
                    }
                if method in ("justin", "random"):
                    recipe["run"]["task"] = "evaluate"
                    for field in (
                        "train_seeds",
                        "train_episodes",
                        "checkpoint_every_episodes",
                        "final_evaluation",
                        "train_ticks",
                        "budget_boundary",
                    ):
                        recipe.pop(field, None)
                    recipe["evaluation"].pop("checkpoints", None)
                target = (
                    ROOT
                    / "experiments/v2"
                    / scale
                    / f"{method + '-shared' if method in ('base', 'isl') else method}-{seed}"
                    / "experiment.yaml"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    "# Generated by scripts/build_v2_experiments.py; Large is validation only.\n"
                    + yaml.safe_dump(recipe, sort_keys=False)
                )
                if method in ("base", "isl"):
                    independent = deepcopy(recipe)
                    independent["algorithm"] = (
                        f"../../../../configs/algorithms/v2-{method}-independent-ddqn.yaml"
                    )
                    target = (
                        ROOT
                        / f"experiments/v2/{scale}"
                        / f"{method}-independent-{seed}/experiment.yaml"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        "# Generated by scripts/build_v2_experiments.py; Large is validation only.\n"
                        + yaml.safe_dump(independent, sort_keys=False)
                    )
        print(
            f"{scale}: 80 pinned worlds, 17 recipes (12 Base/ISL, 3 Morris, Justin, Random)"
        )


if __name__ == "__main__":
    build()
