"""Versioned Vega-Lite specifications: sample SD, seed deduplication and filters."""


def comparison_spec(worlds, *, paired=None):
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"},
        "params": [
            {
                "name": "condition_filter",
                "value": "all",
                "bind": {
                    "input": "select",
                    "options": ["all", "normal", "low", "medium", "high"],
                },
            },
            {
                "name": "world_filter",
                "value": "all",
                "bind": {
                    "input": "select",
                    "options": ["all", *sorted(set(worlds) - {"all"})],
                },
            },
            {
                "name": "methods",
                "select": {"type": "point", "fields": ["series"]},
                "bind": "legend",
            },
        ],
        "transform": [
            {
                "filter": "datum.selection === '${string:selection}' && datum.phase === '${string:phase}'"
            },
            {
                "filter": "datum.condition === condition_filter && datum.world === world_filter"
            },
            {
                "calculate": "datum.complete_inventory ? datum['${field:value}'] : null",
                "as": "observed_value",
            },
            {
                "window": [{"op": "row_number", "as": "revision_rank"}],
                "sort": [
                    {"field": "complete_inventory", "order": "descending"},
                    {"field": "revision", "order": "descending"},
                ],
                "groupby": ["series", "seed", "episode"],
            },
            {"filter": "datum.revision_rank === 1"},
            # Rule rows duplicated in multiple learners have seed='reference'.
            {
                "aggregate": [
                    {"op": "mean", "field": "observed_value", "as": "value"},
                    {"op": "max", "field": "planned_seeds", "as": "planned"},
                ],
                "groupby": ["series", "seed", "episode"],
            },
            {
                "aggregate": [
                    {"op": "mean", "field": "value", "as": "mean"},
                    {"op": "stdev", "field": "value", "as": "sample_sd"},
                    {"op": "valid", "field": "value", "as": "actual_seeds"},
                    {"op": "max", "field": "planned", "as": "planned_seeds"},
                ],
                "groupby": ["series", "episode"],
            },
            {
                "calculate": "datum.actual_seeds === datum.planned_seeds ? '完整' : '临时'",
                "as": "coverage",
            },
            {
                "calculate": "datum.actual_seeds > 1 ? datum.mean - datum.sample_sd : null",
                "as": "lower",
            },
            {
                "calculate": "datum.actual_seeds > 1 ? datum.mean + datum.sample_sd : null",
                "as": "upper",
            },
            {"filter": {"param": "methods"}},
        ],
        "facet": {
            "column": {"field": "coverage", "type": "nominal", "title": "统计覆盖"}
        },
        "spec": {
            "encoding": {
                "x": {
                    "field": "episode",
                    "type": "quantitative",
                    "title": "验证 episode（独立最终评估为 0）",
                },
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "title": "方法／研究变体",
                },
            },
            "layer": [
                {
                    "mark": {"type": "area", "opacity": 0.15},
                    "encoding": {
                        "y": {
                            "field": "lower",
                            "type": "quantitative",
                            "title": "${string:title}",
                        },
                        "y2": {"field": "upper"},
                    },
                },
                {
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "y": {
                            "field": "mean",
                            "type": "quantitative",
                            "title": "${string:title}",
                        },
                        "tooltip": [
                            {"field": "series"},
                            {"field": "episode"},
                            {"field": "mean"},
                            {"field": "sample_sd", "title": "样本 SD"},
                            {"field": "actual_seeds"},
                            {"field": "planned_seeds"},
                            {"field": "coverage"},
                        ],
                    },
                },
            ],
        },
    }

    spec["transform"].insert(2, {"filter": "isValid(datum.seed)"})
    if paired is not None:
        index = next(i for i, t in enumerate(spec["transform"]) if "aggregate" in t)
        spec["transform"][index:index] = [
            {
                "calculate": "datum.observation === 'base' ? datum.observed_value : null",
                "as": "baseline_value",
            },
            {
                "joinaggregate": [
                    {"op": "mean", "field": "baseline_value", "as": "baseline"},
                    {"op": "valid", "field": "baseline_value", "as": "baseline_count"},
                ],
                "groupby": ["method", "seed", "episode", "panel_id"],
            },
            {
                "filter": "datum.observation === 'sl' && isValid(datum.panel_id) && datum.baseline_count === 1"
            },
            {
                "calculate": "datum.observed_value - datum.baseline"
                if paired == "difference"
                else "datum.baseline > 0 ? (datum.baseline-datum.observed_value)/datum.baseline : null",
                "as": "observed_value",
            },
        ]
    method_selection = spec["params"].pop()
    spec["transform"] = [
        t for t in spec["transform"] if t.get("filter") != {"param": "methods"}
    ]
    spec["spec"]["layer"][1]["params"] = [method_selection]
    for layer in spec["spec"]["layer"]:
        layer["transform"] = [{"filter": {"param": "methods"}}]
    return spec


def entity_spec(
    entities, metrics, kind, *, seeds=(), worlds=(), episodes=(), methods=()
):
    ratios = {
        "machines": ["processing_fraction", "available_processing_fraction"],
        "vehicles": ["moving_fraction", "loaded_fraction"],
        "buffers": ["mean_resident"],
    }[kind]
    choices = ratios + metrics
    group = ["tick", "case", "world", "condition", "method", "episode", "selection"]
    aggregate = [
        {"op": "sum", "field": "value", "as": "numerator"},
        {"op": "sum", "field": "observed_ticks", "as": "observed"},
        {"op": "sum", "field": "available_ticks", "as": "available"},
    ]
    calculation = {
        "calculate": "selected_metric === 'available_processing_fraction' ? (datum.available > 0 ? datum.numerator/datum.available : null) : indexof(['processing_fraction','moving_fraction','loaded_fraction','mean_resident'], selected_metric) >= 0 ? (datum.observed > 0 ? datum.numerator/datum.observed : null) : datum.numerator",
        "as": "total",
    }
    params = [
        {
            "name": "selected_metric",
            "value": ratios[0],
            "bind": {"input": "select", "options": choices},
        }
    ]
    transforms = [{"filter": "datum.kind === '${string:kind}'"}]
    for field, values in (
        ("seed", seeds),
        ("world", worlds),
        ("episode", episodes),
        ("series", methods),
    ):
        values = sorted(set(values))
        if values:
            params.append(
                {
                    "name": "selected_" + field,
                    "value": values[-1] if field == "episode" else values[0],
                    "bind": {"input": "select", "options": values},
                }
            )
            transforms.append({"filter": f"datum.{field} === selected_{field}"})
    transforms += [
        {
            "window": [{"op": "row_number", "as": "revision_rank"}],
            "sort": [{"field": "revision", "order": "descending"}],
            "groupby": [
                "seed",
                "series",
                "entity",
                "tick",
                "case",
                "world",
                "condition",
                "episode",
                "selection",
            ],
        },
        {"filter": "datum.revision_rank === 1"},
        {
            "calculate": "selected_metric === 'processing_fraction' || selected_metric === 'available_processing_fraction' ? datum.processing : selected_metric === 'moving_fraction' ? datum.moving : selected_metric === 'loaded_fraction' ? datum.loaded : selected_metric === 'mean_resident' ? datum.resident_ticks : datum[selected_metric]",
            "as": "value",
        },
    ]
    encoding = {
        "x": {"field": "tick", "type": "quantitative", "title": "实际物理 tick"},
        "y": {
            "field": "total",
            "type": "quantitative",
            "title": "所选指标；灰线为该类资源汇总",
        },
        "detail": [{"field": field} for field in group if field != "tick"],
        "tooltip": [
            {"field": field}
            for field in ("world", "case", "condition", "episode", "tick", "total")
        ],
    }
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"},
        "params": params,
        "transform": transforms,
        "layer": [
            {
                "transform": [{"aggregate": aggregate, "groupby": group}, calculation],
                "mark": {"type": "line", "color": "#666666", "point": True},
                "encoding": encoding,
            },
            {
                "params": [
                    {
                        "name": "entities",
                        "select": {"type": "point", "fields": ["entity"]},
                        "bind": "legend",
                    }
                ],
                "transform": [
                    {"aggregate": aggregate, "groupby": group + ["entity"]},
                    calculation,
                ],
                "mark": {"type": "line", "point": True},
                "encoding": {
                    **encoding,
                    "color": {
                        "field": "entity",
                        "type": "nominal",
                        "scale": {"domain": sorted(set(entities))},
                        "title": "点击选择实体（Shift 多选）",
                    },
                    "opacity": {
                        "condition": {"param": "entities", "empty": False, "value": 1},
                        "value": 0,
                    },
                    "tooltip": encoding["tooltip"] + [{"field": "entity"}],
                },
            },
        ],
    }


def mapped(spec, fields):
    """Resolve W&B field placeholders before using stable local Vega field names."""
    spec["transform"][:0] = [
        {"calculate": "datum['${field:" + field + "}']", "as": field}
        for field in fields
    ]
    return spec
