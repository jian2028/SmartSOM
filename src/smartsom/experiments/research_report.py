"""Rebuild one report per concrete study from its local trial evidence."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from smartsom.experiments.evidence import write_json
from smartsom.telemetry.evaluation_summary import METRICS, across_seeds, hierarchical


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_reports(root, plan):
    studies = defaultdict(list)
    for entry in plan["files"]:
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
            for mode in ("sample", "argmax"):
                case_file = trial / "evaluation" / mode / "cases.jsonl"
                cases = (
                    [json.loads(line) for line in case_file.read_text().splitlines()]
                    if case_file.is_file()
                    else []
                )
                inventory = read(
                    trial / "evaluation" / mode / "case-commits" / "plan.json"
                )
                planned = len(inventory.get("cases", []))
                complete = planned > 0 and len(cases) == planned
                temporary = hierarchical(cases)["metrics"]
                metrics = temporary if complete else {key: None for key in METRICS}
                evaluated = read(trial / "evaluation" / mode / "manifest.json")
                checkpoint = evaluated.get("checkpoint")
                checkpoint = trial / Path(checkpoint).name if checkpoint else None
                row = {
                    "method": method,
                    "seed": seed,
                    "decoder": mode,
                    "status": progress.get("status", "not_started"),
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
                    aggregate[method, mode, metric].append(metrics.get(metric))
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        statistics = [
            {
                "method": m,
                "decoder": d,
                "metric": k,
                **across_seeds(values, len(values)),
            }
            for (m, d, k), values in aggregate.items()
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
            "| Method | Seed | Decoder | Status | Model | Results |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| {r['method']} | {r['seed']} | {r['decoder']} | {r['status']} | {r['checkpoint'] or 'missing'} | {r['result']} |"
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
