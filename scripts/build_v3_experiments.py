"""Derive V3 recipes from the integrated V2 campaign without changing its inputs."""

from copy import deepcopy
from pathlib import Path

import yaml

from smartsom.config.base_diagnostic import generate_scenario
from smartsom.config.codec import digest, primitive
from smartsom.config.scenario_authoring import ScenarioFile

ROOT = Path(__file__).resolve().parents[1]


def build():
    algorithms = {}
    for scale in ("small", "medium", "large"):
        scene = ScenarioFile.model_validate(
            yaml.safe_load((ROOT / f"configs/scenarios/v2/{scale}.yaml").read_text())
        )
        envelope, workload = scene.expand()
        hashes = {}
        for source in sorted(
            (ROOT / "experiments/v2" / scale).glob("*/experiment.yaml")
        ):
            data = yaml.safe_load(source.read_text())
            algorithm_source = (source.parent / data["algorithm"]).resolve()
            name = algorithm_source.stem
            name = (
                name.replace("v2-", "v3-", 1)
                if name.startswith("v2-")
                else "v3-" + name
            )
            name = {
                "v3-justin": "v3-rule",
                "v3-morris-e1a-current-ppo": "v3-central-ppo",
            }.get(name, name)
            target_algorithm = ROOT / "configs/algorithms" / (name + ".yaml")
            if name not in algorithms:
                algorithm = deepcopy(yaml.safe_load(algorithm_source.read_text()))
                algorithm["reward"] = {
                    "target": "smartsom.rewards.completion.create_reward",
                    "params": {
                        "makespan_weight": 1.0,
                        "tardiness_weight": 1.0,
                        "scale": 1000.0,
                        "truncation_fixed_penalty": 10.0,
                        "truncation_unfinished_weight": 10.0,
                    },
                }
                if name == "v3-rule":
                    production = algorithm["production"]
                    production["policies"]["rule"] = production["policies"].pop(
                        production["default"]
                    )
                    production["default"] = "rule"
                protocol = algorithm.setdefault("decision_protocol", {})
                protocol["quality_penalty"] = 1000.0
                actions = algorithm.get("action_representation", {})
                if "dispatcher.pickup" in actions.get("rules", {}):
                    actions["rules"]["dispatcher.pickup"] = [
                        "v2_NPF",
                        "v2_HPOF",
                        "v2_EDSF",
                        "v2_LPCL",
                    ]
                    # Keep V2 semantic rule IDs/feature slots to preserve the network.
                    actions["custom_rules"]["v2_HPOF"] = {
                        "target": "smartsom.representations.v3.rules.create_hpof"
                    }
                    active = {
                        rule for rules in actions["rules"].values() for rule in rules
                    }
                    actions["custom_rules"] = {
                        rule: spec
                        for rule, spec in actions["custom_rules"].items()
                        if rule in active
                    }
                for policy in algorithm["production"]["policies"].values():
                    if policy["kind"] == "learning":
                        params = policy["learner"].setdefault("params", {})
                        params.update(
                            gamma=1.0,
                            learning_reward_scale=1.0,
                            drain_deadline_terminal=True,
                        )
                target_algorithm.write_text(yaml.safe_dump(algorithm, sort_keys=False))
                algorithms[name] = algorithm
            data["algorithm"] = (
                "../../../../configs/algorithms/" + target_algorithm.name
            )
            entry = source.parent.name.replace("justin-", "rule-").replace(
                "morris-", "central-"
            )
            target = ROOT / "experiments/v3" / scale / entry / "experiment.yaml"
            # Keep each recipe's explicitly authored limit, without a shared source.
            if target.exists():
                data["drain_ticks"] = yaml.safe_load(target.read_text())["drain_ticks"]
            drain = data["drain_ticks"]
            # Preserve V2 scenarios/seeds; re-pin world hashes for the new deadline.
            data["episode_end"] = "drain_deadline"
            data["drain_ticks"] = drain
            for case in data["test"]["cases"]:
                key = (case["condition"], case["seed"], drain)
                if key not in hashes:
                    world = generate_scenario(
                        envelope.factory,
                        workload,
                        case["seed"],
                        case["condition"],
                        drain,
                    )
                    hashes[key] = digest(primitive(world))
                case["scenario_sha256"] = hashes[key]
            if data.get("periodic_evaluation", {}).get("selection", "last") != "last":
                data["periodic_evaluation"]["cost_metric"] = "learning_cost"
            data["tracking"].update(mode="online", group="v3-" + scale)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "# Generated from V2 by scripts/build_v3_experiments.py.\n"
                "# Large and Central are authored only; do not launch them in this campaign.\n"
                + yaml.safe_dump(data, sort_keys=False)
            )
        print(scale, "17 recipes; Central and Large excluded from launch selection")


if __name__ == "__main__":
    build()
