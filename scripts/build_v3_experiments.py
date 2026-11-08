"""Derive V3 recipes from the integrated V2 campaign without changing its inputs."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build():
    algorithms = {}
    for scale in ("small", "medium", "large"):
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
            target_algorithm = ROOT / "configs/algorithms" / (name + ".yaml")
            if name not in algorithms:
                algorithm = deepcopy(yaml.safe_load(algorithm_source.read_text()))
                algorithm["reward"] = {
                    "target": "smartsom.rewards.completion.create_reward",
                    "params": {
                        "makespan_weight": 1.0,
                        "tardiness_weight": 1.0,
                        "scale": 1000.0,
                    },
                }
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
                        params.update(gamma=1.0, learning_reward_scale=1.0)
                target_algorithm.write_text(yaml.safe_dump(algorithm, sort_keys=False))
                algorithms[name] = algorithm
            data["algorithm"] = (
                "../../../../configs/algorithms/" + target_algorithm.name
            )
            # Same relative depth: retain the exact V2 scenario, worlds and seeds.
            data["episode_end"] = "all_delivered"
            if data.get("periodic_evaluation", {}).get("selection", "last") != "last":
                data["periodic_evaluation"]["cost_metric"] = "learning_cost"
            data["tracking"].update(mode="online", group="v3-" + scale)
            target = (
                ROOT / "experiments/v3" / scale / source.parent.name / "experiment.yaml"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "# Generated from V2 by scripts/build_v3_experiments.py.\n"
                "# Large and Morris are authored only; do not launch them in this campaign.\n"
                + yaml.safe_dump(data, sort_keys=False)
            )
        print(scale, "17 recipes; Morris and Large excluded from launch selection")


if __name__ == "__main__":
    build()
