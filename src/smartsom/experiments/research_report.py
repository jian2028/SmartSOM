"""Rebuild one report per concrete study from its local trial evidence."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from smartsom.experiments.evidence import write_json
from smartsom.telemetry.evaluation_summary import METRICS, across_seeds, hierarchical


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def unique_cases(cases):
    """Repeated receipts do not create repeated scientific observations."""
    from smartsom.config.codec import digest

    seen, output = {}, []
    for case in cases:
        identity = case.get("evaluation_identity")
        if identity is None:
            output.append(case)
            continue
        key = digest(
            [
                identity,
                case.get("method"),
                case.get("condition"),
                case.get("seed"),
                case.get("action_seed"),
                case.get("selection"),
            ]
        )
        if key in seen:
            if seen[key] != case:
                raise ValueError(
                    "conflicting evaluation receipts for one case identity"
                )
            continue
        seen[key] = case
        output.append(case)
    return output


def evaluation_panels(trial):
    """Keep prespecified checkpoints separate, including unfinished panels."""
    evaluation = trial / "evaluation"
    nodes = read(trial / "authoring.json").get("evaluation", {}).get("checkpoints", [])
    if nodes:
        return [
            (
                node["episode"],
                mode,
                evaluation / f"episode-{node['episode']:08d}" / mode,
            )
            for node in nodes
            for mode in node["modes"]
        ]
    # Older evidence may lack authoring.json but retain its checkpoint panels.
    directories = sorted(
        path
        for path in evaluation.glob("episode-*")
        if path.is_dir() and path.name.removeprefix("episode-").isdigit()
    )
    if directories:
        return [
            (int(path.name.removeprefix("episode-")), mode, path / mode)
            for path in directories
            for mode in ("sample", "argmax")
            if (path / mode).is_dir()
        ]
    return [(None, mode, evaluation / mode) for mode in ("sample", "argmax")]


def write_reports(root, plan, state=None):
    studies = defaultdict(list)
    seen = set()
    for entry in plan["files"]:
        if entry["child"] in seen:
            continue
        seen.add(entry["child"])
        studies[entry.get("study") or "."].append(entry)
    for study, entries in studies.items():
        destination = Path(root) / study
        output = destination / "report"
        output.mkdir(parents=True, exist_ok=True)
        rows, aggregate = [], defaultdict(list)
        for entry in entries:
            trial = Path(root) / entry["child"]
            manifest = read(trial / "manifest.json")
            progress = read(trial / "progress.json")
            method = trial.parent.name
            seed = manifest.get("seed", trial.name.removeprefix("seed-"))
            for checkpoint_episode, mode, panel in evaluation_panels(trial):
                case_file = panel / "cases.jsonl"
                cases = (
                    [json.loads(line) for line in case_file.read_text().splitlines()]
                    if case_file.is_file()
                    else []
                )
                cases = unique_cases(cases)
                inventory = read(panel / "case-commits" / "plan.json")
                planned = len(inventory.get("cases", []))
                complete = planned > 0 and len(cases) == planned
                temporary = hierarchical(cases)["metrics"]
                metrics = temporary if complete else {key: None for key in METRICS}
                evaluated = read(panel / "manifest.json")
                checkpoint = evaluated.get("checkpoint")
                checkpoint = (
                    trial
                    / ("checkpoints" if checkpoint_episode is not None else "")
                    / Path(checkpoint).name
                    if checkpoint
                    else None
                )
                row = {
                    "method": method,
                    "seed": seed,
                    "decoder": mode,
                    "checkpoint_episode": checkpoint_episode,
                    "status": progress.get(
                        "status",
                        (state or {})
                        .get("files", {})
                        .get(entry.get("id"), {})
                        .get("status", "not_started"),
                    ),
                    "actual_cases": len(cases),
                    "planned_cases": planned or None,
                    "evaluation_complete": complete,
                    "result": str(trial.relative_to(destination)),
                    "checkpoint": str(checkpoint.relative_to(destination))
                    if checkpoint and checkpoint.is_file()
                    else None,
                    "checkpoint_sha256": evaluated.get("checkpoint_sha256"),
                    "wandb_url": progress.get("tracking", {}).get("url"),
                    "physical_ticks": progress.get("physical_ticks"),
                    "wall_seconds": progress.get("wall_seconds"),
                    **metrics,
                    **{f"provisional_{key}": value for key, value in temporary.items()},
                }
                rows.append(row)
                for metric in METRICS:
                    aggregate[method, mode, checkpoint_episode, metric].append(
                        metrics.get(metric)
                    )
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        statistics = [
            {
                "method": m,
                "decoder": d,
                "checkpoint_episode": episode,
                "metric": k,
                **across_seeds(values, len(values)),
            }
            for (m, d, episode, k), values in aggregate.items()
        ]
        write_json(
            output / "summary.json",
            {"schema": "smartsom.research-report/v1", "statistics": statistics},
        )
        lines = [
            f"# {destination.name}",
            "",
            "Local experiment evidence; missing values are not zero. Engineering runs do not establish research superiority.",
            "",
            "| Method | Seed | Decoder | Episode | Status | Model | Results |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| {r['method']} | {r['seed']} | {r['decoder']} | {r['checkpoint_episode'] if r['checkpoint_episode'] is not None else 'selected'} | {r['status']} | {r['checkpoint'] or 'missing'} | {r['result']} |"
            for r in rows
        )
        lines.extend(
            [
                "",
                "Metrics: [CSV](metrics.csv), [seed coverage and statistics](summary.json).",
                "",
                "Weights: repeat mean → equal worlds → equal conditions → equal training seeds. Sample and argmax remain separate. Complete J is missing for undrained cases.",
            ]
        )
        (output / "report.md").write_text("\n".join(lines) + "\n")
        if destination != Path(root):
            write_json(
                destination / "manifest.json",
                {
                    "schema": "smartsom.research-study/v1",
                    "parent_run_id": plan["run_id"],
                    "source": plan["source"],
                    "files": entries,
                },
            )
