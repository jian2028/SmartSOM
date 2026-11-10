"""Build the single maintained V3 campaign through existing component inputs."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Eight templates per length. Each column is a permutation of A-H, and every
# row contains distinct types. These are fixed products, not runtime RNG draws.
ROUTES = (
    "EBA FCH BDC HAF DEB CFG GHD AGE "
    "CBGD AEBG BDAE GHEA HGDF DCFB EFCH FAHC "
    "BCFED HAECB DGHFC EFAGH ADBHG GBDAE FECBA CHGDF "
    "HDFEGC GBHAED CEBFAG AGCDHF EFGCBH DCABFE FADHCB BHEGDA"
).split()

DECAY = {
    "small": dict(
        dispatcher=300000, machine=260000, pre=210000, post=210000, input=60000
    ),
    "medium": dict(
        dispatcher=900000, machine=680000, pre=600000, post=600000, input=150000
    ),
    # Authored only: extrapolated by workload size, not measured or approved to run.
    "large": dict(
        dispatcher=1800000, machine=1360000, pre=1200000, post=1200000, input=300000
    ),
}


def write_yaml(path, data, comment=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comment + yaml.safe_dump(data, sort_keys=False))


def build():
    for scale in ("small", "medium", "large"):
        scenario = yaml.safe_load(
            (ROOT / "configs/scenarios/v2" / f"{scale}.yaml").read_text()
        )
        scenario["factory"]["factory_id"] = f"v3_{scale}"
        scenario["factory"]["name"] = f"V3 {scale}"
        scenario["jobs"]["routes"] = {
            "families": [list(route) for route in ROUTES],
            "repeat_first_probability": 0.0,
        }
        write_yaml(ROOT / "configs/scenarios/v3" / f"{scale}.yaml", scenario)
        written = set()
        for source in sorted(
            (ROOT / "experiments/v2" / scale).glob("*/experiment.yaml")
        ):
            if not source.parent.name.endswith("-101"):
                continue
            data = yaml.safe_load(source.read_text())
            algorithm_source = (source.parent / data["algorithm"]).resolve()
            name = algorithm_source.stem.removeprefix("v2-")
            name = {"justin": "rule", "morris-e1a-current-ppo": "central-ppo"}.get(
                name, name
            )
            algorithm_path = ROOT / "configs/algorithms/v3" / scale / f"{name}.yaml"
            if name not in written:
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
                algorithm.setdefault("decision_protocol", {})["quality_penalty"] = (
                    1000.0
                )
                production = algorithm["production"]
                if name == "rule":
                    production["policies"]["rule"] = production["policies"].pop(
                        production["default"]
                    )
                    production["default"] = "rule"
                    production["policies"]["rule"]["implementation"]["params"] = {
                        "processing_mode": "normal"
                    }
                actions = algorithm.get("action_representation", {})
                if "dispatcher.pickup" in actions.get("rules", {}):
                    actions["rules"]["dispatcher.pickup"] = [
                        "v2_NPF",
                        "v2_HPOF",
                        "v2_EDSF",
                        "v2_LPCL",
                    ]
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
                for policy in production["policies"].values():
                    if policy["kind"] != "learning":
                        continue
                    params = policy["learner"].setdefault("params", {})
                    params.update(
                        gamma=1.0,
                        learning_reward_scale=1.0,
                        drain_deadline_terminal=True,
                    )
                    if ".dqn." not in policy["learner"]["target"]:
                        continue
                    params.update(
                        update_trigger="transitions",
                        update_every=128,
                        update_amount="ratio",
                        updates_per_transition=0.0078125,
                        batch_size=256,
                        warmup_transitions=1024,
                        target_update_clock="network_gradient_steps",
                        target_update_interval=100,
                        replay_capacity=10000,
                        learning_rate=0.0003,
                        exploration="linear",
                        epsilon=0.3,
                        epsilon_final=0.05,
                        epsilon_decay_transitions=DECAY[scale]["dispatcher"],
                    )
                    decay = dict(DECAY[scale])
                    if "independent" in name:
                        decay.update(
                            {
                                role: 30000 if scale == "small" else 32000
                                for role in decay
                                if role != "input"
                            }
                        )
                    policy["role_params"] = {
                        role: {"epsilon_decay_transitions": count}
                        for role, count in decay.items()
                    }
                write_yaml(algorithm_path, algorithm)
                written.add(name)
            entry = (
                source.parent.name.replace("justin-", "rule-")
                .replace("morris-", "central-")
                .removesuffix("-101")
            )
            if entry == "rule":
                entry = "rule-normal"
            target = ROOT / "experiments/v3" / scale / entry / "experiment.yaml"
            data["scenario"] = f"../../../../configs/scenarios/v3/{scale}.yaml"
            data["algorithm"] = f"../../../../configs/algorithms/v3/{scale}/{name}.yaml"
            # Preserve a later explicitly authored calibration value. Never reuse
            # old V3's values as if the changed route distribution were calibrated.
            data["drain_ticks"] = (
                yaml.safe_load(target.read_text())["drain_ticks"]
                if target.exists()
                else 10000
            )
            data["drain_frozen"] = True
            data["episode_end"] = "drain_deadline"
            data["validation"] = data.pop("periodic_evaluation", {})
            data["validation"]["seeds"] = 50
            data.pop("periodic_evaluation_cases", None)
            data.pop("test_cases", None)
            data["test"] = {"seeds": 50}
            data["train_seeds"] = (
                list(range(101, 106)) if data["run"]["task"] == "train" else [101]
            )
            data["concurrency"]["evaluation_processes"] = 20
            if scale in ("small", "medium"):
                data["concurrency"].update(
                    max_concurrent=3 if scale == "small" else 5, evaluation_cpus=20
                )
                data["resources"].update(
                    cpus=12 if scale == "small" else 24,
                    memory_gib=32 if scale == "small" else 80,
                )
                data["resources"]["guard"]["max_worker_rss_gib"] = (
                    16 if scale == "small" else 32
                )
            if data["validation"].get("selection", "last") != "last":
                data["validation"]["cost_metric"] = "learning_cost"
                data["validation"]["selection"] = "argmax_delivery_cost"
            if data["run"].get("split") == "periodic_evaluation":
                data["run"]["split"] = "validation"
            data["run"]["tune"] = "off"
            data["tracking"].update(mode="online", group=f"v3-{scale}", variant=entry)
            # Two-environment candidate for DQN throughput qualification only.
            data["sampling"] = (
                dict(num_envs=2, processes=2, fragment_ticks=256)
                if "ddqn" in name
                else dict(num_envs=1, processes=0, fragment_ticks=256)
            )
            write_yaml(
                target,
                data,
                "# V3 campaign: deadline provenance and launch selection in experiments/v3/README.md.\n",
            )
        rule = ROOT / "experiments/v3" / scale / "rule-normal/experiment.yaml"
        for mode in ("slow", "fast"):
            algorithm = yaml.safe_load(
                (ROOT / "configs/algorithms/v3" / scale / "rule.yaml").read_text()
            )
            algorithm["production"]["policies"]["rule"]["implementation"]["params"][
                "processing_mode"
            ] = mode
            write_yaml(
                ROOT / "configs/algorithms/v3" / scale / f"rule-{mode}.yaml", algorithm
            )
            recipe = yaml.safe_load(rule.read_text())
            recipe["algorithm"] = (
                f"../../../../configs/algorithms/v3/{scale}/rule-{mode}.yaml"
            )
            recipe["tracking"]["variant"] = f"rule-{mode}"
            target = (
                ROOT / "experiments/v3" / scale / f"rule-{mode}" / "experiment.yaml"
            )
            if target.exists():
                recipe["drain_ticks"] = yaml.safe_load(target.read_text())[
                    "drain_ticks"
                ]
            write_yaml(target, recipe)
        if scale != "large":
            calibration = yaml.safe_load(rule.read_text())
            calibration["test"] = {"seeds": list(range(71001, 71009))}
            calibration["validation"] = {
                "milestones": [],
                "selection": "last",
                "modes": ["argmax"],
            }
            calibration["tracking"]["group"] = f"v3-{scale}-calibration"
            calibration["drain_ticks"] = 10000
            write_yaml(
                ROOT / "experiments/v3/calibration" / scale / "experiment.yaml",
                calibration,
                "# Rule calibration only: 32 worlds, independent of validation/test.\n",
            )


if __name__ == "__main__":
    build()
