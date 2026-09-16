"""Offline reports derived from persisted trace and metrics, never a simulator.

HTML uses inline SVG/JavaScript only. Publication formats import matplotlib on
request. Display playback traverses recorded events; it is not replay validation.
"""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path

from smartsom.experiments.catalog import CURRENT_SCHEMAS, contained_path, read_json
from smartsom.experiments.report_evaluation import evaluation_data

_PAYLOAD_DIRECTORIES = {
    "__pycache__",
    "source",
    "checkpoints",
    "checkpoint",
    "initialized-model",
    "imported-model",
    "imports",
    "_dependencies",
}


def _training_context(path, root):
    """Keep resumed cumulative ledgers distinct instead of concatenating episodes."""
    context = {}
    snapshot = path / "resolved_training.json"
    if snapshot.is_file():
        value = read_json(snapshot)
        if value.get("schema") == "smartsom.resolved-training/v1":
            context["training_seed"] = value["resolved"]["run"]["seed"]
    for owner in (path, *path.parents):
        if not owner.is_relative_to(root):
            break
        metadata = owner / "run.json"
        if not metadata.is_file():
            continue
        record = read_json(metadata)
        if record.get("schema") != "smartsom.experiment/v2" or record.get(
            "kind"
        ) not in {"training", "train_evaluate"}:
            continue
        attempts = [*record.get("attempts", []), record]
        context.update(
            training_seed=record.get("seed"),
            training_run=owner.relative_to(root).as_posix(),
        )
        for index, attempt in enumerate(attempts):
            training = attempt.get("paths", {}).get("training")
            if training and path == contained_path(owner, training):
                context.update(
                    attempt=index,
                    current_attempt=index == len(attempts) - 1,
                    attempt_status=attempt.get("status"),
                    ledger_scope="Cumulative ledger at this attempt; overlapping episodes are not pooled.",
                )
                break
        break
    return context


def _jsonl(path, limit):
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError(f"report input exceeds 256 MiB; select a smaller run: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"report record must be an object: {path}")
                rows.append(value)
                if len(rows) > limit:
                    raise ValueError(
                        f"report row limit exceeded; select a smaller run: {path}"
                    )
    return rows


def _job_mapping(workload):
    if "workload" in workload:
        workload = workload["workload"]
    return {
        operation["operation_id"]: job["job_id"]
        for order in workload.get("orders", ())
        for job in order["jobs"]
        for operation in job["operations"]
    }


def timeline(
    trace: list[dict], jobs: dict[str, str] | None = None, *, initial_state=None
) -> dict:
    """Derive actual processing/travel/wait intervals from observed transitions."""
    if trace and trace[0].get("schema") == "smartsom.production-tick/v1":
        return grid_timeline(trace, initial_state=initial_state)
    jobs = jobs or {}
    events, intervals, active = [], [], {}
    previous_tick = -1

    def start(key, row, kind, resource, job, label):
        if key in active:
            raise ValueError(f"trace starts an already active interval: {key}")
        active[key] = {
            "start": row["simulation_time"],
            "start_sequence": row["sequence"],
            "kind": kind,
            "resource": resource,
            "job": job,
            "label": label,
        }

    def close(key, row, completed=True):
        if key in active:
            intervals.append(
                active.pop(key)
                | {
                    "end": row["simulation_time"],
                    "end_sequence": row["sequence"],
                    "complete": completed,
                }
            )

    for index, raw in enumerate(trace):
        row = dict(raw)
        tick = row.get("simulation_time")
        sequence = row.get("sequence")
        if type(tick) is not int or tick < previous_tick or sequence != index:
            raise ValueError(
                "trace must have ordered integer ticks and contiguous sequences"
            )
        previous_tick = tick
        kind = row.get("kind", "unknown")
        action = row.get("action", {})
        operation = action.get("operation_id", "")
        job = row.get("job_id", jobs.get(operation, ""))
        machine = row.get("machine_id")
        resource = f"machine:{machine}" if machine is not None else ""
        label = operation or kind
        if kind in {"dispatch", "resume"}:
            close(("pause", operation), row)
            start(("processing", operation), row, "processing", resource, job, label)
        elif kind in {"pause", "complete"}:
            close(("processing", operation), row)
            if kind == "pause":
                start(("pause", operation), row, "paused", resource, job, label)
        elif kind == "breakdown":
            start(("down", machine), row, "down", resource, "", "breakdown")
        elif kind == "repair":
            close(("down", machine), row)
        trip = row.get("trip")
        if trip is not None:
            job, resource = trip["job_id"], f"agv:{trip['agv_id']}"
            key = ("trip", trip["agv_id"], trip["transport_sequence"])
            label = f"{job} · trip {trip['transport_sequence']}"
            if kind == "empty_start":
                start(key, row, "empty travel", resource, job, label)
            elif kind == "pickup":
                close(key, row)
            elif kind == "loaded_start":
                start(key, row, "loaded travel", resource, job, label)
            elif kind == "arrival":
                close(key, row)
            elif kind == "wait_for_unload":
                close(key, row)
                start(key, row, "waiting to unload", resource, job, label)
            elif kind == "delivery":
                close(key, row)
        transfer = row.get("transfer")
        if transfer is not None:
            job = transfer["job_id"]
        events.append(
            {
                "sequence": sequence,
                "time": tick,
                "kind": kind,
                "resource": resource,
                "job": job,
                "label": label,
                "detail": row,
            }
        )
    if trace:
        for key in tuple(active):
            close(key, trace[-1], completed=False)
    return {
        "events": events,
        "intervals": intervals,
        "resources": sorted({item["resource"] for item in intervals}),
        "jobs": sorted({item["job"] for item in events if item["job"]}),
        "end_time": max((event["time"] for event in events), default=0),
    }


def grid_timeline(trace, *, initial_state=None):
    """Derive intervals from committed grid records without running simulation."""
    from smartsom.trace.production import Playback

    events = [
        {
            "sequence": 0,
            "time": 0,
            "kind": "state_committed",
            "resource": "",
            "job": "",
            "label": "Initial state",
            "detail": {"tick": 0},
        }
    ]
    intervals, active = [], {}
    for tick, row in enumerate(trace, 1):
        Playback._validate(row, tick)
        start_sequence = len(events) - 1
        before = (
            trace[tick - 2]["state"]
            if tick > 1
            else initial_state
            or {
                "machines": {
                    key: {"down": state["down"], "status": "IDLE", "job": None}
                    for key, state in row["state"]["machines"].items()
                },
                "stations": {},
            }
        )
        if tick == 1 and initial_state is None:
            for event in row["events"]:
                if event["tick"] == tick and event["kind"] in ("breakdown", "repair"):
                    before["machines"][event["machine"]]["down"] = (
                        event["kind"] == "repair"
                    )
        for event in row["events"]:
            resource = next(
                (
                    f"{kind}:{event[key]}"
                    for key, kind in (
                        ("machine", "machine"),
                        ("agv", "agv"),
                        ("station", "quality"),
                    )
                    if key in event
                ),
                "",
            )
            events.append(
                {
                    "sequence": len(events),
                    "time": event["tick"],
                    "kind": event["kind"],
                    "resource": resource,
                    "job": event.get("job", ""),
                    "label": event["kind"],
                    "detail": event,
                }
            )
        events.append(
            {
                "sequence": len(events),
                "time": tick,
                "kind": "state_committed",
                "resource": "",
                "job": "",
                "label": "Committed state",
                "detail": {"tick": tick},
            }
        )
        states = {}
        for key, state in before["machines"].items():
            status = "down" if state["down"] else state["status"].lower()
            if status != "idle":
                states[f"machine:{key}"] = (status, state["job"] or "")
        for event in row["events"]:
            if event["kind"] == "move":
                vehicle = row["state"]["agvs"][event["agv"]]
                states[f"agv:{event['agv']}"] = (
                    "loaded travel" if vehicle["job"] else "empty travel",
                    vehicle["job"] or "",
                )
        for key, state in before["stations"].items():
            if state["status"] == "INSPECTING":
                states[f"quality:{key}"] = ("inspection", ", ".join(state["batch"]))
        # The previous boundary owns [tick-1, tick). Starts happen at its left
        # edge; completions and outages at the right edge belong to the next tick.
        for event in row["events"]:
            if event["kind"] in ("processing_started", "processing_completed"):
                states[f"machine:{event['machine']}"] = ("processing", event["job"])
            elif event["kind"] in ("inspection_started", "inspection_completed"):
                states[f"quality:{event['station']}"] = (
                    "inspection",
                    ", ".join(event["jobs"]),
                )
        for resource in set(active) | set(states):
            value = states.get(resource)
            previous = active.get(resource)
            if previous and (previous["kind"], previous["job"]) != value:
                intervals.append(active.pop(resource))
            if value is not None:
                if resource not in active:
                    active[resource] = {
                        "resource": resource,
                        "kind": value[0],
                        "job": value[1],
                        "label": value[0],
                        "start": tick - 1,
                        "end": tick,
                        "start_sequence": start_sequence,
                        "end_sequence": len(events) - 1,
                        "complete": True,
                    }
                active[resource].update(end=tick, end_sequence=len(events) - 1)
    if trace:
        final = trace[-1]["state"]
        for resource, interval in active.items():
            role, key = resource.split(":", 1)
            if role == "machine":
                state = final["machines"][key]
                kind = "down" if state["down"] else state["status"].lower()
                interval["complete"] = (kind, state["job"] or "") != (
                    interval["kind"],
                    interval["job"],
                )
            elif role == "quality":
                interval["complete"] = final["stations"][key]["status"] != "INSPECTING"
    intervals.extend(active.values())
    return {
        "events": events,
        "intervals": intervals,
        "resources": sorted({i["resource"] for i in intervals}),
        "jobs": sorted({e["job"] for e in events if e["job"]}),
        "end_time": len(trace),
    }


def _series(directory, limit):
    series = []
    ledger = directory / "episodes.jsonl"
    if ledger.is_file():
        episodes = _jsonl(ledger, limit)
        for field, label in (
            ("return", "Episode raw return"),
            ("makespan", "Completed episode makespan"),
        ):
            points = [
                [row["episode"], row[field]]
                for row in episodes
                if isinstance(row.get(field), (int, float))
                and not isinstance(row[field], bool)
                and math.isfinite(row[field])
            ]
            if points:
                series.append({"label": label, "x_label": "episode", "points": points})
    metrics = directory / "learner_metrics.jsonl"
    if metrics.is_file():
        values = {}
        for row in _jsonl(metrics, limit):
            for name, value in row.get("metrics", {}).items():
                if any(
                    word in name.lower()
                    for word in ("loss", "entropy", "kl", "explained", "learning_rate")
                ):
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    ):
                        values.setdefault(name, []).append([row["update"], value])
        series.extend(
            {"label": key, "x_label": "learner update", "points": points}
            for key, points in sorted(values.items())
        )
    return series


def load_report_data(source: str | Path, *, max_rows: int = 100_000) -> dict:
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f"report source must be an evidence directory: {root}")
    records, evaluations, models, count = [], [], set(), 0
    evaluated_runs = {}
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(
            n for n in names if not n.startswith(".") and n not in _PAYLOAD_DIRECTORIES
        )
        path = Path(directory)
        if "run.json" in files:
            metadata = read_json(path / "run.json")
            if metadata.get("schema") == "smartsom.evaluation/v1":
                evaluation = evaluation_data(path, root)
                evaluations.append(evaluation)
                models.update(evaluation["models"])
                for row in evaluation["runs"]:
                    evaluated_runs[contained_path(path, row["run_dir"])] = row
            elif metadata.get("schema") == "smartsom.experiment/v2" and metadata.get(
                "kind"
            ) in {"training", "train_evaluate"}:
                from smartsom.experiments.packaging import model_locator

                for selection in ("last", "best"):
                    if (path / "checkpoints" / f"{selection}.json").is_file() or (
                        path / "checkpoints" / selection
                    ).exists():
                        models.add(str(model_locator(path, checkpoint=selection)))
        production = (
            "run.json" in files
            and metadata.get("schema") == "smartsom.production-run/v1"
        )
        if not production and not any(
            name in files
            for name in ("trace.jsonl", "learner_metrics.jsonl", "episodes.jsonl")
        ):
            continue
        summary = read_json(path / "summary.json") if "summary.json" in files else {}
        manifest = read_json(path / "manifest.json") if "manifest.json" in files else {}
        if (
            "run.json" in files
            and read_json(path / "run.json").get("schema")
            == "smartsom.production-run/v1"
        ):
            metadata = read_json(path / "run.json")
            manifest = {
                "provider": metadata["inputs"]["algorithm"]["provider"],
                "source": metadata["source"],
            }
            summary = {
                "status": metadata["status"],
                "tick": metadata["last_tick"],
                "makespan": metadata["last_tick"]
                if metadata["status"] == "completed"
                else None,
            }
        jobs = (
            _job_mapping(read_json(path / "realized_instance.json"))
            if "realized_instance.json" in files
            else {}
        )
        trace = (
            _jsonl(path / "trace.jsonl", max_rows - count)
            if "trace.jsonl" in files
            else []
        )
        count += len(trace)
        relative = path.relative_to(root).as_posix()
        context = _training_context(path, root)
        label = relative if relative != "." else root.name
        evaluated = evaluated_runs.get(path)
        if evaluated:
            label = f"{evaluated['case_id']} · {evaluated['algorithm_id']} · evaluation replication {evaluated['replication']}"
            if evaluated["checkpoint_sha256"]:
                label += f" · training seed {evaluated['training_seed']} · model {evaluated['checkpoint_sha256'][:12]}"
        if "training_seed" in context:
            label += f" · training seed {context['training_seed']}"
        if "attempt" in context:
            label += f" · attempt {context['attempt']:03d} ({'current' if context['current_attempt'] else 'retained'})"
        records.append(
            {
                "id": relative,
                "label": label,
                "training": context,
                "evaluation": evaluated,
                "provider": manifest.get("provider", ""),
                "summary": summary,
                "source": manifest.get("source", {}),
                "series": _series(path, max_rows),
                **timeline(
                    trace,
                    jobs,
                    initial_state=metadata.get("initial_state") if production else None,
                ),
            }
        )
        # Failure traces are retained by training even when successful episode
        # traces are only represented by auditable ledgers. Show only saved traces.
        for failure_path in sorted(path.glob("episode_*_failure.json")):
            failure = read_json(failure_path)
            failure_trace = failure.get("trace", [])
            count += len(failure_trace)
            if count > max_rows:
                raise ValueError("report row limit exceeded; select a smaller run")
            record = failure.get("record", {})
            records.append(
                {
                    "id": failure_path.relative_to(root).as_posix(),
                    "label": f"{label} · episode {record.get('episode', '?')} · {record.get('end_reason', 'failed')}",
                    "provider": manifest.get("provider", ""),
                    "training": context,
                    "summary": {
                        "status": "failed",
                        "end_reason": record.get("end_reason"),
                        "return": record.get("return"),
                    },
                    "source": manifest.get("source", {}),
                    "series": [],
                    **timeline(
                        failure_trace,
                        _job_mapping(failure.get("input", {}).get("workload", {})),
                    ),
                }
            )
    if not records and evaluations:
        records.append(
            {
                "id": ".",
                "label": "Evaluation overview (no persisted trace)",
                "provider": "",
                "summary": {},
                "source": {},
                "series": [],
                **timeline([]),
            }
        )
    if not records:
        raise ValueError("no recorded trace or training metrics found in report source")
    metadata = read_json(root / "run.json") if (root / "run.json").is_file() else {}
    return {
        "schema": "smartsom.offline-report/v1",
        "title": metadata.get("name", root.name),
        "source": str(root),
        "runs": records,
        "evaluations": evaluations,
        "model_references": sorted(models),
        "note": "Display of recorded evidence. Event playback does not perform replay validation.",
    }


def _destination(source, destination):
    root, target = Path(source).resolve(), Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if target.resolve().is_relative_to(root):
        current = root / "run.json"
        if (
            not current.is_file()
            or read_json(current).get("schema") not in CURRENT_SCHEMAS
        ):
            raise ValueError("report destination must be outside historical evidence")
        if not target.resolve().is_relative_to(root / "reports"):
            raise ValueError(
                "derived reports inside a v2 run must use its reports directory"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def build_report(
    source: str | Path, destination: str | Path, *, max_rows: int = 100_000
) -> Path:
    data = load_report_data(source, max_rows=max_rows)
    target = _destination(source, destination)
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    text = _HTML.replace("__TITLE__", html.escape(data["title"])).replace(
        "__DATA__", payload
    )
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
    _protect_report_models(data, target)
    return target


def _protect_report_models(data, target):
    from smartsom.experiments.references import protect_model_reference

    for path in data["model_references"]:
        protect_model_reference(Path(path), target)


def export_figure(
    source: str | Path,
    destination: str | Path,
    *,
    run_id: str | None = None,
    resource: str | None = None,
    job: str | None = None,
    dpi: int = 180,
) -> Path:
    """Export a recorded timeline (or training curve) using a headless renderer."""
    target = Path(destination)
    if target.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("figure destination must end in .png, .svg or .pdf")
    data = load_report_data(source)
    runs = (
        [row for row in data["runs"] if row["id"] == run_id]
        if run_id is not None
        else [row for row in data["runs"] if row["intervals"]]
    )
    if not runs and run_id is None:
        runs = data["runs"]
    if not runs:
        raise ValueError(f"unknown report run: {run_id}")
    row = runs[0]
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError(
            "figure export requires uv sync --locked --extra reports"
        ) from exc
    intervals = [
        i
        for i in row["intervals"]
        if (resource is None or i["resource"] == resource)
        and (job is None or i["job"] == job)
    ]
    resources = sorted({i["resource"] for i in intervals})
    if (resource is not None or job is not None) and not intervals:
        raise ValueError("no timeline intervals match the selected filter")
    figure = Figure(figsize=(12, max(3, min(30, 1.6 + len(resources) * 0.5))))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    if intervals:
        for item in intervals:
            color = _color(item["job"] or item["kind"])
            bars = axes.barh(
                resources.index(item["resource"]),
                item["end"] - item["start"],
                left=item["start"],
                height=0.6,
                color=color,
                alpha=0.45 if item["kind"] != "processing" else 0.9,
                hatch="//"
                if not item["complete"]
                else {
                    "empty travel": "//",
                    "waiting to unload": "..",
                    "paused": "xx",
                    "down": "xx",
                }.get(item["kind"]),
            )
            if (item["end"] - item["start"]) / max(row["end_time"], 1) > 0.055:
                label = item["label"].split(" · ")[0]
                if item["kind"] == "processing":
                    label = label.rsplit("/", 1)[-1]
                text = axes.text(
                    item["start"] + 0.3,
                    resources.index(item["resource"]),
                    label,
                    fontsize=7,
                    va="center",
                    clip_on=True,
                )
                text.set_clip_path(bars.patches[0])
        axes.set_yticks(range(len(resources)), resources)
        axes.invert_yaxis()
        axes.set_xlabel("Simulation time (ticks)")
        axes.grid(axis="x", alpha=0.2)
        labels = sorted({item["job"] for item in intervals if item["job"]})
        if labels:
            axes.legend(
                handles=[Patch(color=_color(label), label=label) for label in labels],
                title="Jobs",
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                fontsize=8,
            )
    elif row["series"]:
        series = row["series"][0]
        axes.plot(*zip(*series["points"]))
        axes.set_xlabel(series["x_label"])
        axes.set_ylabel(series["label"])
    else:
        axes.text(
            0.5, 0.5, "No persisted intervals or numeric training series", ha="center"
        )
        axes.set_axis_off()
    axes.set_title(f"{data['title']} · {row['provider'] or row['label']}")
    figure.text(
        0.01,
        0.01,
        f"Recorded evidence · status: {row['summary'].get('status', 'unknown')}",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    target = _destination(source, destination)
    with target.open("xb") as stream:
        figure.savefig(stream, format=target.suffix[1:].lower(), dpi=dpi)
    _protect_report_models(data, target)
    return target


def _color(label):
    palette = (
        "#2563eb",
        "#0d9488",
        "#7c3aed",
        "#d97706",
        "#db2777",
        "#0891b2",
        "#65a30d",
        "#ea580c",
    )
    value = 0
    for character in label:
        value = (31 * value + ord(character)) & 0xFFFFFFFF
    return palette[value % len(palette)]


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · SmartSOM</title>
<style>
:root{color-scheme:light;--ink:#152437;--muted:#516378;--line:#dce4eb;--accent:#0b766e}
*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;background:#f4f7fa;color:var(--ink);margin:0}
main{max-width:1440px;margin:32px auto;padding:0 28px}h1{font-size:30px;margin:8px 0}h2{font-size:19px;margin:0 0 14px}.eyebrow{letter-spacing:.12em;font-size:12px;color:var(--accent);font-weight:700}
.muted{color:var(--muted)}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}label{display:flex;gap:8px;align-items:center}select,button,input{font:inherit}button,select{padding:7px 10px;background:white;border:1px solid #b9c7d4;border-radius:6px;color:var(--ink)}button{cursor:pointer}button:hover{background:#ecfdf5}select{max-width:700px}input[type=range]{flex:1;min-width:180px}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;text-transform:uppercase;color:var(--muted)}.scroll{overflow:auto;max-height:440px}svg{width:100%;min-width:600px;display:block}.chart{overflow:auto}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{padding:12px 16px;border-left:3px solid var(--accent);background:#f7fafb}.metric strong{font-size:22px;display:block}pre{font-size:12px;white-space:pre-wrap;max-height:220px;overflow:auto;background:#f8fafc;padding:10px;border-radius:6px}.selected{background:#e8f4f1}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px;border-radius:2px}details{margin-top:12px}button:focus-visible,select:focus-visible{outline:3px solid #7dd3fc}.footnote{font-size:12px}.wide{width:100%}@media print{body{background:white}main{max-width:none;margin:0}.controls,button,input{display:none}.card{break-inside:avoid}.scroll{max-height:none}details{display:none}}
</style></head><body><main><div class="eyebrow">SMARTSOM / RECORDED EXPERIMENT</div><h1>__TITLE__</h1>
<p class="muted">Offline report · all data and rendering are included in this file.</p>
<section class="card" id="evaluationCard"><h2>Evaluation coverage and paired outcomes</h2><div id="evaluations"></div></section>
<section class="card"><h2>Runs and outcomes</h2><div class="scroll"><table><thead><tr><th>Run / episode</th><th>Provider</th><th>Status</th><th>Makespan</th><th>Raw return</th></tr></thead><tbody id="summary"></tbody></table></div></section>
<section class="card"><div class="controls"><label>Run <select id="run"></select></label></div><div id="metrics" class="metrics"></div><details><summary>Recorded source</summary><pre id="source"></pre></details></section>
<section class="card"><h2>Recorded event timeline</h2><div class="controls"><label>Resource <select id="resource"></select></label><label>Job <select id="job"></select></label><button id="svgExport">Save SVG</button><button id="pngExport">Save PNG</button><button id="pdfExport">Print / Save PDF</button></div>
<div class="controls"><button id="zoomOut" aria-label="Zoom out">−</button><button id="zoomIn" aria-label="Zoom in">+</button><button id="zoomReset">Fit timeline</button><span id="zoomLevel" aria-live="polite">1×</span><span class="footnote muted">Zoom expands the time axis; scroll horizontally to pan.</span></div>
<p id="noTrace" class="muted"></p><div class="chart" id="ganttViewport"><svg id="gantt" role="img" aria-label="Resource Gantt chart"></svg></div><div id="legend" class="legend"></div>
<div class="controls"><button id="first" aria-label="First event">⏮</button><button id="previous" aria-label="Previous event">◀</button><button id="play">Play</button><button id="next" aria-label="Next event">▶</button><input id="cursor" type="range" min="0" max="0" value="0" aria-label="Recorded event"><label>Speed <select id="speed"><option value="750">Slow</option><option value="250" selected>Normal</option><option value="60">Fast</option></select></label><span id="position" aria-live="polite"></span></div>
<p class="footnote muted">Playback follows recorded event order, including events at the same tick. Dashed intervals indicate incomplete evidence. This display is not simulator replay validation.</p>
<details open><summary>Current event</summary><pre id="eventDetail"></pre></details>
<p class="footnote muted">Most recent 120 matching events up to the current cursor.</p><div class="scroll"><table><thead><tr><th>Sequence</th><th>Tick</th><th>Event</th><th>Resource</th><th>Job / operation</th></tr></thead><tbody id="events"></tbody></table></div></section>
<section class="card"><h2>Training curves</h2><label>Metric <select id="series"></select></label><div class="chart"><svg id="curve" role="img" aria-label="Training metric curve"></svg></div><p id="curveNote" class="footnote muted"></p></section>
<footer class="muted footnote">Generated from local evidence. Failed attempts remain visible; missing makespan is not zero. Authoritative traces, checkpoint digests and validation reports remain separate.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
"use strict";
const D=JSON.parse(document.getElementById("data").textContent),$=id=>document.getElementById(id),NS="http://www.w3.org/2000/svg";
let current=0,cursor=0,timer=null,zoom=1;const palette=["#2563eb","#0d9488","#7c3aed","#d97706","#db2777","#0891b2","#65a30d","#ea580c"];
function el(tag,text,parent){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(parent)parent.append(n);return n}
function svgEl(tag,attrs,text,parent){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;if(parent)parent.append(n);return n}
function options(id,values,all=false){const s=$(id);s.replaceChildren();if(all){const o=el("option","All",s);o.value=""}for(const[value,label]of values){const o=el("option",label,s);o.value=value}}
function color(job){let h=0;for(const c of job)h=(Math.imul(h,31)+c.codePointAt(0))|0;return palette[(h>>>0)%palette.length]}
function stop(){if(timer!==null)clearInterval(timer);timer=null;$("play").textContent="Play"}
function matches(i){return (!$("resource").value||i.resource===$("resource").value)&&(!$("job").value||i.job===$("job").value)}
function draw(){const r=D.runs[current],s=$("gantt"),ev=r.events[cursor],tick=ev?ev.time:0,seq=ev?ev.sequence:-1;
const list=r.intervals.filter(matches),resources=[...new Set(list.map(i=>i.resource))].sort(),left=140,right=24,baseWidth=Math.max(600,$("ganttViewport").clientWidth||1150),width=left+right+(baseWidth-left-right)*zoom,height=Math.max(130,resources.length*34+64),max=Math.max(1,r.end_time),x=t=>left+t/max*(width-left-right);
s.replaceChildren();s.style.width=zoom===1?"100%":`${width}px`;$("zoomLevel").textContent=`${zoom}×`;s.setAttribute("viewBox",`0 0 ${width} ${height}`);svgEl("rect",{width,height,fill:"white"},undefined,s);
for(let i=0;i<=10*zoom;i++){const t=max*i/(10*zoom),xx=x(t);svgEl("line",{x1:xx,y1:12,x2:xx,y2:height-34,stroke:"#e2e8f0"},undefined,s);svgEl("text",{x:xx,y:height-13,"text-anchor":"middle","font-size":12,fill:"#516378"},Number(t.toFixed(1)),s)}
resources.forEach((resource,i)=>svgEl("text",{x:left-10,y:i*34+33,"text-anchor":"end","font-size":12,fill:"#152437"},resource,s));
for(const item of list){if(item.start_sequence>seq)continue;const y=resources.indexOf(item.resource)*34+16,end=item.end_sequence<=seq?item.end:Math.min(item.end,tick),w=Math.max(1,x(end)-x(item.start));const rect=svgEl("rect",{x:x(item.start),y,width:w,height:23,rx:3,fill:item.kind==="down"?"#94a3b8":color(item.job||item.kind),opacity:item.kind==="processing"?.9:.48,stroke:!item.complete?"#111827":"none","stroke-dasharray":!item.complete?"4 3":"none"},undefined,s);svgEl("title",{},`${item.label} / ${item.kind}\n${item.start} → ${item.end}${item.complete?"":" (partial)"}`,rect);if(w>46)svgEl("text",{x:x(item.start)+4,y:y+16,"font-size":11,fill:"#102032"},item.label.slice(0,Math.max(3,Math.floor(w/7))),s)}
svgEl("line",{x1:x(tick),y1:10,x2:x(tick),y2:height-32,stroke:"#0f172a","stroke-width":2},undefined,s);
$("legend").replaceChildren();for(const job of [...new Set(list.map(i=>i.job).filter(Boolean))]){const n=el("span",undefined,$("legend")),sw=el("span",undefined,n);sw.className="swatch";sw.style.background=color(job);n.append(document.createTextNode(job))}
$("cursor").max=Math.max(0,r.events.length-1);$("cursor").value=cursor;$("position").textContent=ev?`event ${cursor+1} / ${r.events.length} · tick ${tick}`:"No recorded events";$("eventDetail").textContent=ev?JSON.stringify(ev.detail,null,2):"No physical trace was persisted for this selection.";
$("noTrace").textContent=r.events.length?"":"Training metrics are available below. A successful episode ledger is not presented as a physical trace.";
$("events").replaceChildren();for(const e of r.events.filter(matches).filter(e=>e.sequence<=seq).slice(-120)){const tr=el("tr",undefined,$("events"));if(e.sequence===seq)tr.className="selected";for(const v of[e.sequence,e.time,e.kind,e.resource,e.job||e.label])el("td",String(v),tr);tr.onclick=()=>{stop();cursor=e.sequence;draw()}}
}
function curve(){const row=D.runs[current].series[Number($("series").value)],s=$("curve");s.replaceChildren();s.setAttribute("viewBox","0 0 1150 280");svgEl("rect",{width:1150,height:280,fill:"white"},undefined,s);if(!row){$("curveNote").textContent="No persisted training metrics for this selection.";return}
const pts=row.points,xmin=Math.min(...pts.map(p=>p[0])),xmax=Math.max(...pts.map(p=>p[0])),ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1])),x=v=>90+(v-xmin)/Math.max(1,xmax-xmin)*1030,y=v=>230-(v-ymin)/(ymax-ymin||1)*190;
for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,yy=y(v);svgEl("line",{x1:90,y1:yy,x2:1120,y2:yy,stroke:"#e2e8f0"},undefined,s);svgEl("text",{x:80,y:yy+4,"text-anchor":"end","font-size":11,fill:"#516378"},Number(v.toPrecision(5)),s)}
svgEl("polyline",{points:pts.map(p=>`${x(p[0])},${y(p[1])}`).join(" "),fill:"none",stroke:"#0b766e","stroke-width":2},undefined,s);for(const p of pts){const dot=svgEl("circle",{cx:x(p[0]),cy:y(p[1]),r:2,fill:"#0b766e"},undefined,s);svgEl("title",{},`${row.x_label} ${p[0]}: ${p[1]}`,dot)}svgEl("text",{x:90,y:254,"font-size":12},xmin,s);svgEl("text",{x:1120,y:254,"text-anchor":"end","font-size":12},xmax,s);svgEl("text",{x:575,y:274,"text-anchor":"middle","font-size":12},row.x_label,s);$("curveNote").textContent=`${row.label}. Values come from saved metrics; raw return and learner-scaled optimization metrics use different units.`}
function select(){stop();zoom=1;$("ganttViewport").scrollLeft=0;const r=D.runs[current];cursor=Math.max(0,r.events.length-1);options("resource",r.resources.map(v=>[v,v]),true);options("job",r.jobs.map(v=>[v,v]),true);options("series",r.series.map((v,i)=>[i,v.label]));$("metrics").replaceChildren();for(const[key,value]of Object.entries(r.summary)){if(value===null||typeof value==="object")continue;if(!["status","makespan","environment_steps","agent_steps","learner_updates","completed_episodes","failed_episodes","end_reason"].includes(key))continue;const n=el("div",undefined,$("metrics"));n.className="metric";el("small",key.replaceAll("_"," "),n);el("strong",String(value),n)}$("source").textContent=JSON.stringify({source:r.source,training:r.training??null,evaluation:r.evaluation??null},null,2);draw();curve()}
function download(blob,name){const a=document.createElement("a"),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function svgText(){const copy=$("gantt").cloneNode(true);copy.setAttribute("xmlns",NS);const box=copy.viewBox.baseVal;copy.setAttribute("width",box.width);copy.setAttribute("height",box.height);return new XMLSerializer().serializeToString(copy)}
$("svgExport").onclick=()=>download(new Blob([svgText()],{type:"image/svg+xml"}),"smartsom-timeline.svg");
$("pngExport").onclick=()=>{const text=svgText(),url=URL.createObjectURL(new Blob([text],{type:"image/svg+xml"})),im=new Image();im.onload=()=>{const canvas=document.createElement("canvas");canvas.width=im.width*2;canvas.height=im.height*2;const ctx=canvas.getContext("2d");ctx.scale(2,2);ctx.drawImage(im,0,0);canvas.toBlob(blob=>{if(blob)download(blob,"smartsom-timeline.png")});URL.revokeObjectURL(url)};im.onerror=()=>URL.revokeObjectURL(url);im.src=url};$("pdfExport").onclick=()=>window.print();
function reportTable(parent,headers,rows){const wrap=el("div",undefined,parent);wrap.className="scroll";const table=el("table",undefined,wrap),head=el("tr",undefined,el("thead",undefined,table));for(const h of headers)el("th",h,head);const body=el("tbody",undefined,table);for(const values of rows){const tr=el("tr",undefined,body);for(const value of values)el("td",value===null||value===undefined?"—":String(value),tr)}}
$("evaluationCard").hidden=!D.evaluations.length;
for(const evaluation of D.evaluations){const section=el("section",undefined,$("evaluations"));el("h3",evaluation.id+" · "+evaluation.status,section);if(evaluation.error)el("p",`Failure at ${evaluation.stage??"unknown stage"}: ${typeof evaluation.error==="string"?evaluation.error:JSON.stringify(evaluation.error)}`,section);el("p",evaluation.note,section);reportTable(section,["Case / algorithm","Model digest / training seed","Completed / requested","Missing / duplicates","Failure reasons","Complete-case mean makespan"],evaluation.coverage.map(r=>[r.case_id+" / "+r.algorithm_id,(r.checkpoint_sha256??"baseline")+" / "+(r.training_seed??"unknown"),r.completed+" / "+(r.requested??"unknown"),r.missing+" / "+r.duplicates,JSON.stringify(r.failure_reasons),r.complete_case_mean_makespan]));reportTable(section,["Case / replication","Baseline","Training seed","Pair status","Makespan delta (model − baseline)"],evaluation.pairs.map(r=>[r.case_id+" / "+r.replication,r.baseline,r.training_seed,r.status,r.makespan_delta]));if(evaluation.input_coverage){el("h4","Input coverage",section);el("p",evaluation.input_coverage.interpretation,section);const coverage=evaluation.input_coverage;reportTable(section,["Case","Same training factory","Same training workload"],(coverage.cases??[]).map(r=>[r.case_id,r.same_training_factory,r.same_training_workload]));const training=coverage.training_history??{},validation=coverage.validation??{};el("p",`Training history: ${training.status??"unavailable"}; recorded episodes: ${training.episodes??"unknown"}; overlapping evaluation worlds: ${training.overlapping_evaluation_worlds?.length??"unknown"}.`,section);if(training.all_training_samples_covered===false)el("p","Recorded episodes cover only the retained training history; full sample coverage is unverified.",section);el("p",`Validation inputs: ${validation.status??"unavailable"}; overlapping evaluation worlds: ${validation.overlapping_evaluation_worlds?.length??"unknown"}.`,section);const detail=el("details",undefined,section);el("summary","Recorded coverage details",detail);el("pre",JSON.stringify(coverage,null,2),detail)}else el("p","Input overlap metadata was not recorded for this evaluation.",section);if(!evaluation.pairs.length)el("p","No baseline pair was declared in this evaluation.",section);for(const warning of evaluation.warnings)el("p",warning,section)}
options("run",D.runs.map((r,i)=>[i,`${r.provider?r.provider+" · ":""}${r.label}`]));for(const r of D.runs){const tr=el("tr",undefined,$("summary"));for(const v of[r.label,r.provider,r.summary.status??"unknown",r.summary.makespan??"—",r.summary.return??"—"])el("td",String(v),tr)}
$("zoomIn").onclick=()=>{zoom=Math.min(8,zoom*2);draw()};$("zoomOut").onclick=()=>{zoom=Math.max(1,zoom/2);draw()};$("zoomReset").onclick=()=>{zoom=1;$("ganttViewport").scrollLeft=0;draw()};
$("run").onchange=()=>{current=Number($("run").value);select()};$("resource").onchange=draw;$("job").onchange=draw;$("series").onchange=curve;$("first").onclick=()=>{stop();cursor=0;draw()};$("previous").onclick=()=>{stop();cursor=Math.max(0,cursor-1);draw()};$("next").onclick=()=>{stop();cursor=Math.min(D.runs[current].events.length-1,cursor+1);draw()};$("cursor").oninput=()=>{stop();cursor=Number($("cursor").value);draw()};$("speed").onchange=()=>{if(timer!==null){stop();$("play").click()}};$("play").onclick=()=>{if(timer!==null){stop();return}const end=D.runs[current].events.length-1;if(end<0)return;if(cursor>=end)cursor=0;$("play").textContent="Pause";draw();timer=setInterval(()=>{cursor++;draw();if(cursor>=end)stop()},Number($("speed").value))};select();
</script></body></html>"""
