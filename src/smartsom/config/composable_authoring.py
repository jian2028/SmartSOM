"""Classified authoring files and explicit, reviewable legacy migration."""

from pathlib import Path

import yaml

from smartsom.config.codec import ConfigurationError, primitive
from smartsom.config.experiment import load_config
from smartsom.config.experiment_v3 import read_document


def scaffold(destination, name="example"):
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in name
    ):
        raise ConfigurationError(
            "name must contain letters, digits, underscore or hyphen"
        )
    destination = Path(destination).resolve()
    source = Path(__file__).resolve().parents[3] / "configs" / "test"
    planned = {}

    def put(relative, data):
        planned[destination / relative] = yaml.safe_dump(data, sort_keys=False)

    scenario = read_document(source / "scenarios/template1_static.yaml")
    for field, category in (("factory", "factories"), ("workload", "workloads")):
        original = (source / "scenarios" / scenario[field]).resolve()
        target = f"{category}/{name}.yaml"
        planned[destination / target] = original.read_text()
        scenario[field] = f"../{target}"
    put(f"scenarios/{name}.yaml", scenario)
    rule_names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "shortest_path",
    }
    for role in ("machine", "buffer", "dispatcher", "mover"):
        for kind in (rule_names[role], "new_ppo", "new_dqn"):
            put(
                f"policies/{name}_{role}_{kind}.yaml",
                read_document(source / f"policies/{role}_{kind}.yaml"),
            )
    for kind in ("ppo", "dqn"):
        put(
            f"algorithms/{name}_{kind}.yaml",
            read_document(source / f"algorithms/{kind}.yaml"),
        )
    for matching in ("global_optimal", "priority_greedy"):
        put(
            f"rules/{name}_pickup_{matching}.yaml",
            read_document(source / f"rules/pickup_{matching}.yaml"),
        )
    for kind in ("all_rules", "train_machine_ppo", "train_machine_dqn"):
        composition = read_document(source / f"compositions/{kind}.yaml")
        composition["pickup_matching"] = f"../rules/{name}_pickup_global_optimal.yaml"
        for group in composition["groups"].values():
            original = Path(group["policy"]).name
            group["policy"] = "../policies/" + name + "_" + original
        put(f"compositions/{name}_{kind}.yaml", composition)
        example = read_document(
            source
            / f"runs/{'evaluate_all_rules' if kind == 'all_rules' else kind}.yaml"
        )
        example["scenario"] = f"../scenarios/{name}.yaml"
        example["composition"] = f"../compositions/{name}_{kind}.yaml"
        example["output"] = {"root": "../../runs", "name": name + "_" + kind}
        if "training" in example:
            example["training"]["parameters"] = (
                f"../algorithms/{name}_{example['training']['algorithm']}.yaml"
            )
        put(f"runs/{name}_{'evaluate' if kind == 'all_rules' else kind}.yaml", example)
    collisions = [str(path) for path in planned if path.exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite: " + ", ".join(collisions))
    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {
        "configs": str(destination),
        "files": [str(p) for p in planned],
        "next": f"smartsom show-config --config {destination / 'runs' / (name + '_train_machine_ppo.yaml')}",
    }


def migration_preview(source, output, *, total_ticks=None, ticks_per_update=None):
    old = load_config(source)
    if old.schema_id.endswith("/v3"):
        raise ConfigurationError("configuration is already v3")
    output = Path(output).resolve()
    if output.parent.name != "runs" or output.parent.parent.name != "configs":
        raise ConfigurationError(
            "v3 authoring experiment must be placed under configs/runs"
        )
    provider = read_document(old.algorithm.source)["algorithm"]["provider"]
    learning = provider in (
        "rllib.resource_ppo",
        "rllib.ppo",
        "sb3.maskable_ppo",
    )
    if learning and (total_ticks is None or ticks_per_update is None):
        raise ConfigurationError(
            "legacy steps are not physical ticks; supply both --total-ticks and --ticks-per-update"
        )
    root = Path(__file__).resolve().parents[3] / "configs" / "test"
    central = provider in ("rllib.ppo", "sb3.maskable_ppo")
    composition = (
        "central_sb3_ppo"
        if provider == "sb3.maskable_ppo"
        else "central_rllib_ppo"
        if central
        else "train_machine_ppo"
        if learning
        else "all_rules"
    )
    result = {
        "schema": "smartsom.experiment-config/v3",
        "scenario": old.scenario,
        "composition": str(root / "compositions" / (composition + ".yaml")),
        "scenario_overrides": primitive(old.scenario_overrides),
        "seed": old.seed,
        "validation": primitive(old.validation),
        "evaluation": primitive(old.evaluation),
        "output": primitive(old.output),
    }
    if learning:
        result["training"] = {
            "mode": "central" if central else "resource",
            "backend": "sb3" if provider == "sb3.maskable_ppo" else "rllib",
            "algorithm": "ppo",
            "groups": ["central" if central else "machine"],
            "parameters": str(root / "algorithms/ppo.yaml"),
            "total_ticks": total_ticks,
            "ticks_per_update": ticks_per_update,
        }
    return {
        "original_preserved": str(Path(source).resolve()),
        "output": str(output),
        "configuration": result,
        "differences": [
            "four decision roles replace the combined AGV decision",
            "legacy checkpoints are not mapped; learning starts fresh",
            "resource migration initializes Machine only with rule partners; edit composition to select more groups",
            "explicit physical tick budget replaces legacy steps",
        ],
    }


def migrate(source, output, *, preview=False, **budgets):
    result = migration_preview(source, output, **budgets)
    if not preview:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            yaml.safe_dump(result["configuration"], stream, sort_keys=False)
    return result
