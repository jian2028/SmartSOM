"""Execution-only selection; never rewrite the authored scientific seed list."""

import argparse

import yaml

from smartsom.config.codec import ConfigurationError, _UniqueLoader, safe_yaml_load


class EntryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        ids = list(getattr(namespace, self.dest, None) or [])
        selectors = list(getattr(namespace, "batch_selectors", None) or [])
        setattr(namespace, self.dest, ids + [values])
        namespace.batch_selectors = selectors + [None]


class SelectAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        selectors = list(getattr(namespace, "batch_selectors", None) or [])
        if not selectors and getattr(namespace, "_allow_file_select", False):
            if getattr(namespace, "file_selector", None) is not None:
                raise argparse.ArgumentError(self, "only one file selection is allowed")
            try:
                namespace.file_selector = parse_selector(values)
            except (ValueError, yaml.YAMLError) as exc:
                raise argparse.ArgumentError(self, str(exc)) from exc
            return
        if not selectors or selectors[-1] is not None:
            raise argparse.ArgumentError(
                self, "requires a preceding --id with no --select yet"
            )
        try:
            selectors[-1] = parse_selector(values)
        except (ValueError, yaml.YAMLError) as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        namespace.batch_selectors = selectors


def parse_selector(value):
    from smartsom.config.base_diagnostic import BaseExperiment

    if isinstance(value, str):
        value = yaml.load(value, Loader=_UniqueLoader)
    if not isinstance(value, dict) or set(value) != {"train_seeds"}:
        raise ConfigurationError("--select requires only the train_seeds YAML field")
    # Reuse the schema exactly: integers are counts from 101, lists are identities.
    return {"train_seeds": list(BaseExperiment.expand_seeds(value["train_seeds"]))}


def requests(ids, selectors, first, overrides):
    if selectors is not None and (ids is None or len(selectors) != len(ids)):
        raise ConfigurationError("--select must bind to exactly one preceding --id")
    selectors = selectors if selectors is not None else [None] * len(ids or [])
    local = any(s is not None for s in selectors)
    seed_override = overrides.get("seed") is not None
    for item in overrides.get("sets", []):
        key, _, raw = item.partition("=")
        if key in {"experiment.train_seeds", "experiment.run.seed"}:
            seed_override = True
        elif key == "experiment.run":
            value = safe_yaml_load(raw)
            if isinstance(value, dict) and "seed" in value:
                seed_override = True
    if (local and (first is not None or seed_override)) or (
        first is not None and seed_override
    ):
        raise ConfigurationError(
            "execution seed selection conflicts with first-N/scientific seed overrides"
        )
    return [
        {
            "entry": entry,
            "selector": parse_selector(selector) if selector is not None else None,
        }
        for entry, selector in zip(ids or [], selectors, strict=True)
    ]


def select_tasks(files, requests, first, entry_for_path):
    selected = []
    seen = set()
    for request in requests:
        rows = [row for row in files if entry_for_path(row["path"]) == request["entry"]]
        selector = request["selector"]
        if selector is not None:
            if any(row["compiled"].experiment.run.task != "train" for row in rows):
                raise ConfigurationError(
                    "--select train_seeds requires a training Experiment"
                )
            seeds = selector["train_seeds"]
            for row in rows:
                if not set(seeds) <= set(row["compiled"].experiment.train_seeds):
                    raise ConfigurationError(
                        f"{request['entry']}: selected seed is not in authored train_seeds"
                    )
            rows = [
                row
                for seed in seeds
                for row in rows
                if row["compiled"].experiment.run.seed == seed
            ]
        elif first is not None:
            rows = [
                row
                for row in rows
                if row["compiled"].experiment.run.task != "train"
                or row["compiled"].experiment.run.seed
                in row["compiled"].experiment.train_seeds[:first]
            ]
        for row in rows:
            if row["id"] in seen:
                raise ConfigurationError(f"duplicate selected task: {row['id']}")
            seen.add(row["id"])
            selected.append(row["id"])
    return selected
