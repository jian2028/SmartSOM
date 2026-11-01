"""Explicit saved W&B views from recorded runs; never edits an existing workspace."""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from smartsom.experiments.catalog import resolve_run

HEADLINES = {
    "delivery_fraction": "订单交付比例",
    "drained": "案例排空率",
    "joint_cost": "完整成本 J",
    "qualified_coverage": "合格交付覆盖率",
    "mean_tardiness": "平均拖期",
    "final_passing_rate": "最终合格率",
}
SERVICE = {
    "tardiness_p95": "案例拖期 P95",
    "on_time_coverage": "准时交付覆盖率",
    "mean_flow_time": "平均流动时间",
}


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def records(root):
    """Use local identities and actual events, without querying or starting tracking."""
    root = Path(root)
    found = []
    for path in sorted(root.rglob("tracking/status.json")):
        status = _read(path)
        run = path.parent.parent
        manifest = _read(run / "manifest.json")
        events = run / "tracking/events.jsonl"
        rows = []
        if events.exists():
            lines = events.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise ValueError(f"invalid event journal: {events}") from None
                    # A live writer may not have finished the last line yet.
        if status.get("online_run_id") and status.get("entity"):
            found.append(
                {
                    "root": str(run),
                    "status": status,
                    "manifest": manifest,
                    "events": rows,
                }
            )
    return found


def plan(root):
    groups = defaultdict(list)
    for record in records(root):
        status = record["status"]
        comparison = status.get("comparison", {})
        inputs = record["manifest"].get("inputs", {})
        # Do not accidentally average different maps or workload definitions.
        from smartsom.config.codec import read_document

        frozen = Path(record["root"]) / "inputs/experiment.yaml"
        experiment = read_document(frozen) if frozen.exists() else {}
        cohort = (
            inputs.get("factory"),
            inputs.get("workload"),
            json.dumps(
                {
                    "evaluation": experiment.get("evaluation"),
                    "drain_ticks": record["manifest"].get("drain_ticks"),
                    "split": record["manifest"].get("split"),
                    "validation_cases": experiment.get("validation_cases"),
                    "test_cases": experiment.get("test_cases"),
                },
                sort_keys=True,
            ),
            record["manifest"].get("source", {}).get("implementation_sha256"),
        )
        modes = {
            event.get("selection")
            for event in record["events"]
            if event.get("phase") in {"validation", "evaluation"}
        }
        for mode in sorted(modes - {None}):
            groups[
                (
                    status["entity"],
                    status["project"],
                    comparison.get("map", "custom"),
                    mode,
                    cohort,
                )
            ].append(record)
    if not groups:
        raise ValueError(
            "No recorded online evaluation/validation runs; upload offline segments and retain resolved cloud identities first."
        )
    return groups


def _metric_keys(rows, selection, metric, condition="all"):
    keys = set()
    for record in rows:
        for event in record["events"]:
            if event.get("kind") != "summary" or event.get("selection") != selection:
                continue
            for method, conditions in event.get("methods", {}).items():
                if metric in conditions.get(condition, {}).get("metrics", {}):
                    keys.add(
                        f"{event['phase']}/{method}/{selection}/{condition}/{metric}"
                    )
    return sorted(keys)


def _statistics(rows, selection):
    """Latest summaries only; exact sample SD is computed locally, not SDK bands."""
    from smartsom.telemetry.evaluation_summary import across_seeds
    from smartsom.telemetry.tracking_metadata import method_identity

    groups = defaultdict(dict)
    planned = {}
    for record in rows:
        status = record["status"]
        comparison = status.get("comparison", {})
        experiment = _read(Path(record["root"]) / "manifest.json")
        expected = experiment.get("train_seeds", [])
        # Frozen four-file config is authoritative when manifest omits the list.
        path = Path(record["root"]) / "inputs/experiment.yaml"
        if path.exists():
            from smartsom.config.codec import read_document

            expected = read_document(path).get("train_seeds", expected)
        seed = record["manifest"].get("training_seed", record["manifest"].get("seed"))
        for event in record["events"]:
            if event.get("kind") != "summary" or event.get("selection") != selection:
                continue
            milestone = (event.get("training_progress") or {}).get("episodes", "final")
            for method, conditions in event.get("methods", {}).items():
                key = (
                    method,
                    comparison.get("variant", "custom")
                    if method not in {"rule", "random"}
                    else method,
                    event["phase"],
                    milestone,
                    method_identity(comparison, method),
                )
                planned[key] = max(
                    planned.get(key, 0),
                    len(expected) if method not in {"rule", "random"} else 1,
                )
                # Rule is one reference, not one fictitious training seed per learner.
                identity = seed if method not in {"rule", "random"} else "reference"
                groups[key][identity] = (
                    conditions.get("all", {}) if identity is not None else {}
                )
    output = []
    for key, seeds in sorted(groups.items(), key=lambda item: str(item[0])):
        for metric in HEADLINES:
            values = [
                row.get("metrics", {}).get(metric)
                if row.get("complete_inventory")
                else None
                for row in seeds.values()
            ]
            stats = across_seeds(values, max(planned[key], len(values)))
            output.append(
                {
                    "method": key[0],
                    "variant": key[1],
                    "phase": key[2],
                    "milestone": key[3],
                    "method_identity": key[4],
                    "metric": metric,
                    **stats,
                }
            )
    return output


def _table(statistics):
    lines = [
        "### 本地精确统计快照",
        "",
        "刷新此快照请重新执行 dashboard 命令；实时曲线继续由 W&B 更新。",
        "",
        "|方法／变体|阶段／节点|指标|均值|样本 SD|实际／预定 seed|状态|",
        "|---|---|---|---|---|---|---|",
    ]

    def show(value):
        return (
            "缺失"
            if value is None
            else (f"{value:.6g}" if isinstance(value, float) else str(value)).replace(
                "|", "\\|"
            )
        )

    for row in statistics:
        lines.append(
            "|"
            + "|".join(
                show(value)
                for value in (
                    f"{row['method']}/{row['variant']} [{str(row['method_identity'])[:8]}]",
                    f"{row['phase']}/{row['milestone']}",
                    HEADLINES[row["metric"]],
                    row.get("mean"),
                    row.get("sample_sd"),
                    f"{row.get('actual_seeds', 0)}/{row.get('planned_seeds', 0)}",
                    row.get("status"),
                )
            )
            + "|"
        )
    return "\n".join(lines)


def paired_statistics(rows, selection):
    from smartsom.telemetry.evaluation_summary import paired_sig

    panels = defaultdict(list)
    for record in rows:
        comparison = record["status"].get("comparison", {})
        if comparison.get("observation") not in {"base", "sl"}:
            continue
        grouped = defaultdict(list)
        for event in record["events"]:
            if (
                event.get("selection") != selection
                or "evaluation_identity" not in event
                or event.get("method") in {"rule", "random"}
            ):
                continue
            progress = event["evaluation_identity"].get("training_progress") or {}
            key = (
                comparison.get("method"),
                progress.get("train_seed"),
                progress.get("episodes"),
                event.get("phase"),
            )
            grouped[key].append(event)
        for key, cases in grouped.items():
            panels[key].append((comparison, cases))
    result = []
    for key, entries in panels.items():
        bases = [
            cases
            for comparison, cases in entries
            if comparison["observation"] == "base"
        ]
        for comparison, cases in entries:
            if comparison["observation"] != "sl":
                continue
            outcome = {
                "status": "missing_or_ambiguous_base",
                "sig": None,
                "absolute_sl_minus_base": None,
            }
            if len(bases) == 1:
                try:
                    outcome = paired_sig(bases[0], cases)
                except ValueError as error:
                    outcome = {**outcome, "status": str(error)}
            result.append(
                {
                    "method": key[0],
                    "seed": key[1],
                    "episode": key[2],
                    "variant": comparison.get("variant"),
                    **outcome,
                }
            )
    return result


def car_statistics(rows, selection):
    from smartsom.telemetry.research_comparison import acceleration

    entries = []
    for row in rows:
        comparison = row["status"].get("comparison", {})
        document = _read(Path(row["root"]) / "learning-target.json")
        outcome = document.get("modes", {}).get(selection)
        if outcome is None:
            continue
        signatures = {
            event.get("evaluation_identity", {})
            .get("sl_comparison", {})
            .get("non_sl_sha256")
            for event in row["events"]
            if event.get("selection") == selection
        } - {None}
        signature = next(iter(signatures)) if len(signatures) == 1 else None
        entries.append((comparison, outcome, signature))
    results = []
    for comparison, outcome, signature in entries:
        if comparison.get("observation") != "sl":
            continue
        bases = [
            base
            for metadata, base, other in entries
            if metadata.get("observation") == "base"
            and metadata.get("method") == comparison.get("method")
            and signature is not None
            and other == signature
            and base.get("panel_identity") == outcome.get("panel_identity")
        ]
        result = {"status": "missing_or_ambiguous_base", "ratio": None}
        if len(bases) == 1:
            try:
                result = acceleration(bases[0], outcome)
            except ValueError as error:
                result = {"status": "incompatible", "ratio": None, "reason": str(error)}
        results.append(
            {
                "variant": comparison.get("variant"),
                "panel_identity": outcome.get("panel_identity"),
                **result,
            }
        )
    return {
        "comparisons": results,
        "attainment_evidence": [
            {
                "method": metadata.get("method"),
                "observation": metadata.get("observation"),
                "variant": metadata.get("variant"),
                "result": outcome,
            }
            for metadata, outcome, _ in entries
        ],
    }


def research_evidence(rows, selection):
    """Present existing protocol results verbatim; never invent research formulas."""
    evidence = {"robustness": [], "CAR": [], "controlled_recovery": []}
    car = car_statistics(rows, selection)
    if car["attainment_evidence"]:
        evidence["CAR"].append(car)
    seen = set()
    for row in rows:
        root = Path(row["root"])
        for filename, key in (
            ("research-summary.json", "robustness"),
            ("car-summary.json", "CAR"),
            ("recovery-summary.json", "controlled_recovery"),
        ):
            for path in sorted(root.rglob(filename)):
                if path in seen:
                    continue
                seen.add(path)
                document = _read(path)
                if document.get("selection", selection) != selection:
                    continue
                value = document.get("robustness") if key == "robustness" else document
                if value:
                    evidence[key].append(
                        {
                            "source": str(path.relative_to(root)),
                            "run": row["status"].get("online_run_id"),
                            "result": value,
                        }
                    )
        for event in row["events"]:
            if event.get("selection", selection) != selection:
                continue
            if event.get("kind") in {"car_summary", "car"}:
                evidence["CAR"].append(event)
    return {
        key: values
        or {"status": "not_recorded", "reason": "协议未启用或尚无已记录结果"}
        for key, values in evidence.items()
    }


def build_views(root, *, presets=None):
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as error:
        raise ValueError(
            "Dashboard creation requires the optional dashboard extra: uv sync --extra dashboard"
        ) from error
    from smartsom.telemetry.dashboard_specs import comparison_spec, entity_spec, mapped
    from smartsom.telemetry.experiment_tracking import representative_case

    views = []
    definitions = {}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for (entity, project, scale, mode, cohort), rows in plan(root).items():
        ids = sorted({row["status"]["online_run_id"] for row in rows})
        worlds = {
            str(event["scenario_seed"])
            for row in rows
            for event in row["events"]
            if "scenario_seed" in event
        }
        fields = [
            "phase",
            "selection",
            "condition",
            "world",
            "series",
            "seed",
            "episode",
            "planned_seeds",
            "complete_inventory",
            "revision",
        ]
        spec = mapped(comparison_spec(worlds), fields)
        import hashlib

        fingerprint = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:16]
        preset_key = (entity, fingerprint)
        definitions[preset_key] = spec
        chart_name = (presets or {}).get(preset_key, f"{entity}/smartsom-{fingerprint}")
        phases = sorted(
            {
                event["phase"]
                for row in rows
                for event in row["events"]
                if event.get("kind") == "summary"
            }
        )
        sections = []
        for title, metrics in (("生产总览", HEADLINES), ("交付详情", SERVICE)):
            panels = []
            for metric, label in metrics.items():
                if not _metric_keys(rows, mode, metric):
                    panels.append(
                        wr.MarkdownPanel(markdown=f"### {label}\n尚无记录；不补零。")
                    )
                    continue
                for phase in phases:
                    panels.append(
                        wr.CustomChart(
                            query={
                                "summaryTable": {"tableKey": "research/summary_history"}
                            },
                            chart_name=chart_name,
                            chart_fields={"value": metric, **{f: f for f in fields}},
                            chart_strings={
                                "selection": mode,
                                "phase": phase,
                                "title": label,
                            },
                        )
                    )
            sections.append(
                ws.Section(name=title, is_open=title == "生产总览", panels=panels)
            )
        sections.append(
            ws.Section(
                name="跨 seed 精确统计快照",
                is_open=False,
                panels=[wr.MarkdownPanel(markdown=_table(_statistics(rows, mode)))],
            )
        )
        paired_panels = [
            wr.MarkdownPanel(
                markdown="配对数值核对快照（不支持或身份不一致时不计算）：\n```json\n"
                + json.dumps(
                    paired_statistics(rows, mode), ensure_ascii=False, indent=2
                )
                + "\n```"
            )
        ]
        paired_fields = fields + ["observation", "method", "panel_id"]
        for pair_kind, title in (
            ("difference", "配对完整成本 SL−Base"),
            ("sig", "SIG 相对成本改善"),
        ):
            definition = mapped(
                comparison_spec(worlds, paired=pair_kind), paired_fields
            )
            fingerprint = hashlib.sha256(
                json.dumps(definition, sort_keys=True).encode()
            ).hexdigest()[:16]
            pair_key = (entity, fingerprint)
            definitions[pair_key] = definition
            for phase in phases:
                paired_panels.append(
                    wr.CustomChart(
                        query={
                            "summaryTable": {"tableKey": "research/summary_history"}
                        },
                        chart_name=(presets or {}).get(
                            pair_key, f"{entity}/smartsom-{fingerprint}"
                        ),
                        chart_fields={
                            "value": "joint_cost",
                            **{f: f for f in paired_fields},
                        },
                        chart_strings={
                            "selection": mode,
                            "phase": phase,
                            "title": title,
                        },
                    )
                )
        sections.append(
            ws.Section(name="Base／SL 配对", is_open=False, panels=paired_panels)
        )
        sections.append(
            ws.Section(
                name="鲁棒性／CAR／受控恢复证据",
                is_open=False,
                panels=[
                    wr.MarkdownPanel(
                        markdown="已记录协议结果快照（不重新定义指标）：\n```json\n"
                        + json.dumps(
                            research_evidence(rows, mode), ensure_ascii=False, indent=2
                        )
                        + "\n```"
                    )
                ],
            )
        )
        entity_panels = [
            wr.WeavePanelSummaryTable(table_name=f"research/{mode}/entities")
        ]
        for kind in ("machines", "vehicles", "buffers"):
            entities, metrics = set(), set()
            entity_seeds, entity_worlds, entity_episodes, entity_methods = (
                set(),
                set(),
                set(),
                set(),
            )
            for row in rows:
                for event in row["events"]:
                    if event.get("selection") != mode or not representative_case(event):
                        continue
                    entity_seeds.add(
                        str(
                            row["manifest"].get(
                                "training_seed", row["manifest"].get("seed")
                            )
                        )
                    )
                    entity_worlds.add(str(event.get("scenario_seed")))
                    entity_episodes.add(
                        (event.get("training_progress") or {}).get("episodes", 0)
                    )
                    comparison = row["status"].get("comparison", {})
                    entity_methods.add(
                        f"{event.get('method')}/{comparison.get('variant')} [{comparison.get('method_identity', '')[:8]}]"
                    )
                    for point in event.get("research_metrics", {}).get(
                        "entity_series", []
                    ):
                        for name, values in point.get(kind, {}).items():
                            entities.add(name)
                            metrics.update(
                                k
                                for k, v in values.items()
                                if isinstance(v, (float, int))
                                and not isinstance(v, bool)
                            )
            if entities and metrics:
                metrics.update({"observed_ticks", "available_ticks"})
                entity_fields = [
                    "kind",
                    "entity",
                    "tick",
                    "case",
                    "world",
                    "condition",
                    "method",
                    "episode",
                    "selection",
                    "seed",
                    "series",
                    "revision",
                    *sorted(metrics),
                ]
                entity_definition = mapped(
                    entity_spec(
                        entities,
                        sorted(metrics),
                        kind,
                        seeds=entity_seeds,
                        worlds=entity_worlds,
                        episodes=entity_episodes,
                        methods=entity_methods,
                    ),
                    entity_fields,
                )
                fingerprint = hashlib.sha256(
                    json.dumps(entity_definition, sort_keys=True).encode()
                ).hexdigest()[:16]
                key = (entity, fingerprint)
                definitions[key] = entity_definition
                entity_panels.append(
                    wr.CustomChart(
                        query={
                            "summaryTable": {
                                "tableKey": f"research/{mode}/entity_series"
                            }
                        },
                        chart_name=(presets or {}).get(
                            key, f"{entity}/smartsom-{fingerprint}"
                        ),
                        chart_fields={f: f for f in entity_fields},
                        chart_strings={"kind": kind},
                    )
                )
        sections.append(
            ws.Section(
                name="实体诊断 · 序列仅代表已上传首案例",
                is_open=False,
                panels=entity_panels,
            )
        )
        from smartsom.telemetry.experiment_tracking import cloud_fields, numeric

        keys = set()
        for row in rows:
            for event in row["events"]:
                if event.get("phase") == "train":
                    keys.update(numeric(cloud_fields(event), "train/"))
        for title, prefixes in (
            ("学习诊断", ("train/losses/", "train/diagnostics/")),
            ("资源监测", ("train/resources/",)),
        ):
            panels = [
                wr.LinePlot(
                    title=key, x="train/physical_ticks", y=[key], smoothing_factor=0
                )
                for key in sorted(keys)
                if key.startswith(prefixes)
            ]
            if panels:
                sections.append(ws.Section(name=title, is_open=False, panels=panels))
        cohort_label = (
            str(cohort[0] or "unknown")[:6] + "-" + str(cohort[1] or "unknown")[:6]
        )
        view = ws.Workspace(
            entity=entity,
            project=project,
            name=f"SmartSOM {scale} {mode} {cohort_label} {timestamp}",
            sections=sections,
            auto_generate_panels=False,
        )
        view.runset_settings = ws.RunsetSettings(
            filters=[ws.Metric("ID").isin(ids)],
            pinned_columns=["config:comparison.variant", "config:seed", "run:state"],
        )
        views.append(view)
    return views, definitions


def create_dashboard(source, roots=None):
    root = resolve_run(source, roots)
    # Preflight constructs and validates all local data before any cloud mutation.
    _, definitions = build_views(root)
    from uuid import uuid4

    import wandb

    api = wandb.Api()
    presets = {}
    for (entity, fingerprint), spec in definitions.items():
        presets[(entity, fingerprint)] = api.create_custom_chart(
            entity=entity,
            name=f"smartsom-{fingerprint}-{uuid4().hex[:8]}",
            display_name="SmartSOM research metrics",
            spec_type="vega2",
            access="private",
            spec=spec,
        )
    views, _ = build_views(root, presets=presets)
    result = []
    for view in views:
        view.save()  # Newly constructed views only; no existing layout is loaded.
        restored = type(view).from_url(view.url)
        expected = [(section.name, len(section.panels)) for section in view.sections]
        actual = [(section.name, len(section.panels)) for section in restored.sections]
        if expected != actual:
            raise RuntimeError(
                f"Saved dashboard did not preserve sections/panels: {view.url}; "
                f"expected={expected!r}, actual={actual!r}"
            )
        result.append({"name": view.name, "url": view.url, "verified": True})
    return {"source": str(root), "views": result}
