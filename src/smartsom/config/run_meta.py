"""Unified non-scientific run metadata, compiled facts and display selection."""

import platform
import socket
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator

from smartsom.config.codec import ConfigurationError, StrictModel, read_document

META_SCHEMA = "smartsom.experiment-meta/v1"
META_FILENAME = "meta.yaml"
ETA_SCHEMA = "smartsom.eta/v1"

SHORT_TITLE_LIMIT = 32
LONG_TITLE_LIMIT = 120
SHORT_DESCRIPTION_LIMIT = 200
FULL_DESCRIPTION_LIMIT = 4000

REQUIRED_FORMAL_TASKS = frozenset({"train", "evaluate"})


def _clean_single_line(value: str | None, *, field: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{field} must be single-line text without control characters")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be nonempty when provided")
    if len(cleaned) > limit:
        raise ValueError(f"{field} must be at most {limit} characters")
    return cleaned


def _clean_multiline(value: str | None, *, field: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if any((ord(ch) < 32 and ch != "\n") or ord(ch) == 127 for ch in normalized):
        raise ValueError(f"{field} cannot contain control characters")
    cleaned = normalized.strip()
    if not cleaned:
        raise ValueError(f"{field} must be nonempty when provided")
    if len(cleaned) > limit:
        raise ValueError(f"{field} must be at most {limit} characters")
    return cleaned


class RunMeta(StrictModel):
    """Human- or AI-authored presentation metadata; excluded from scientific identity."""

    short_title: str | None = None
    long_title: str | None = None
    short_description: str | None = None
    full_description: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("short_title")
    @classmethod
    def _valid_short_title(cls, value):
        return _clean_single_line(value, field="short_title", limit=SHORT_TITLE_LIMIT)

    @field_validator("long_title")
    @classmethod
    def _valid_long_title(cls, value):
        return _clean_single_line(value, field="long_title", limit=LONG_TITLE_LIMIT)

    @field_validator("short_description")
    @classmethod
    def _valid_short_description(cls, value):
        return _clean_single_line(
            value, field="short_description", limit=SHORT_DESCRIPTION_LIMIT
        )

    @field_validator("full_description")
    @classmethod
    def _valid_full_description(cls, value):
        return _clean_multiline(
            value, field="full_description", limit=FULL_DESCRIPTION_LIMIT
        )

    @field_validator("tags", mode="before")
    @classmethod
    def _valid_tags(cls, value):
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("meta.tags must be a list of strings")
        cleaned = []
        for item in value:
            tag = _clean_single_line(item, field="meta.tags item", limit=64)
            if tag not in cleaned:
                cleaned.append(tag)
        return tuple(cleaned)

    def is_empty(self) -> bool:
        return (
            self.short_title is None
            and self.long_title is None
            and self.short_description is None
            and self.full_description is None
            and not self.tags
        )

    def authored_dict(self) -> dict:
        data = self.model_dump(mode="json")
        if not data["tags"]:
            data.pop("tags")
        return {key: value for key, value in data.items() if value is not None}


class BatchMetaDocument(RunMeta):
    """Standalone batch-directory metadata file (`meta.yaml`)."""

    schema_id: Literal["smartsom.experiment-meta/v1"] = Field(alias="schema")


def coerce_meta(value) -> RunMeta:
    if value is None:
        return RunMeta()
    if isinstance(value, RunMeta):
        return RunMeta.model_validate(
            {
                "short_title": value.short_title,
                "long_title": value.long_title,
                "short_description": value.short_description,
                "full_description": value.full_description,
                "tags": value.tags,
            }
        )
    if isinstance(value, dict):
        data = dict(value)
        data.pop("schema", None)
        data.pop("effective", None)
        data.pop("auto_summary", None)
        return RunMeta.model_validate(data)
    raise ConfigurationError("meta must be a mapping")


def load_batch_meta(directory: Path) -> RunMeta | None:
    path = Path(directory) / META_FILENAME
    if not path.is_file():
        return None
    raw = read_document(path)
    try:
        doc = BatchMetaDocument.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"{path}: invalid batch meta.yaml: {exc}") from exc
    return coerce_meta(doc)


def launch_batch_meta(directory: Path, paths) -> RunMeta | None:
    """Resolve the submitted directory and, for a single study, its owning directory."""
    from smartsom.config.experiment_layout import batch_directory

    local = load_batch_meta(directory)
    if local is not None:
        return local
    owners = {batch_directory(path) for path in paths}
    if len(owners) == 1:
        owner = owners.pop()
        if owner != directory and directory.is_relative_to(owner):
            return load_batch_meta(owner)
    return None


def merge_meta(*layers) -> RunMeta:
    merged: dict = {
        "short_title": None,
        "long_title": None,
        "short_description": None,
        "full_description": None,
    }
    tags: list[str] = []
    for layer in layers:
        if layer is None:
            continue
        meta = coerce_meta(layer)
        for field in (
            "short_title",
            "long_title",
            "short_description",
            "full_description",
        ):
            value = getattr(meta, field)
            if value is not None:
                merged[field] = value
        for tag in meta.tags:
            if tag not in tags:
                tags.append(tag)
    merged["tags"] = tuple(tags)
    return RunMeta.model_validate(merged)


def first_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def resolved_meta(
    meta: RunMeta | dict | None,
    *,
    fallback_title: str | None = None,
    auto_summary: str | None = None,
) -> dict:
    """Serialize authored metadata alongside deterministic fallback projections."""
    parsed = coerce_meta(meta)
    short_title = parsed.short_title or parsed.long_title or fallback_title
    long_title = parsed.long_title or parsed.short_title or fallback_title
    short_description = (
        parsed.short_description or first_line(parsed.full_description) or auto_summary
    )
    full_description = (
        parsed.full_description or parsed.short_description or auto_summary
    )
    return {
        "short_title": parsed.short_title,
        "long_title": parsed.long_title,
        "short_description": parsed.short_description,
        "full_description": parsed.full_description,
        "tags": list(parsed.tags),
        "auto_summary": auto_summary,
        "effective": {
            "short_title": short_title,
            "long_title": long_title,
            "short_description": short_description,
            "full_description": full_description,
        },
    }


def select_display_text(
    meta_payload: dict | None,
    field: str,
    *,
    fallback: str | None = None,
    authored_only: bool = False,
) -> str | None:
    """Select one configured text field from a resolved meta payload."""
    if field == "none":
        return None
    if field == "default":
        return fallback
    if not isinstance(meta_payload, dict):
        return fallback
    effective = meta_payload.get("effective") or {}
    source = meta_payload if authored_only else effective
    if field == "combined":
        short_title = source.get("short_title") or effective.get("short_title")
        long_title = source.get("long_title") or effective.get("long_title")
        if short_title and long_title and long_title != short_title:
            return f"{short_title} · {long_title}"
        return short_title or long_title or fallback
    if field == "auto_summary":
        return meta_payload.get("auto_summary") or fallback
    if field in {
        "short_title",
        "long_title",
        "short_description",
        "full_description",
    }:
        value = source.get(field)
        if value is not None:
            return value
        if not authored_only:
            return fallback
        if field == "full_description":
            return meta_payload.get("short_description")
        if field == "short_description":
            return first_line(meta_payload.get("full_description"))
        return None
    return fallback


def placeholder_eta(*, updated_at: float | None = None) -> dict:
    """Reserved unified ETA contract; populated by the scheduler ETA engine."""
    return {
        "schema": ETA_SCHEMA,
        "remaining_seconds": None,
        "eta_at_unix": None,
        "confidence": "unavailable",
        "basis": "eta engine pending",
        "breakdown": None,
        "updated_at": updated_at,
    }


def host_identity() -> dict:
    return {
        "name": socket.gethostname(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }


def _unique(values):
    seen = []
    for item in values:
        if item is not None and item not in seen:
            seen.append(item)
    return seen


def _policy_summary(algorithm_doc: dict) -> tuple[str, str, str]:
    policies = algorithm_doc.get("production", {}).get("policies", {})
    learning = [p for p in policies.values() if p.get("kind") == "learning"]
    method, observation = "custom", "custom"
    if not learning:
        default = algorithm_doc.get("production", {}).get("default")
        target = policies.get(default, {}).get("implementation", {}).get("target", "")
        for known in ("rule", "random"):
            if target == f"smartsom.policies.{known}.create_policy":
                method, observation = known, "base"
    elif len(learning) == 1:
        target = learning[0].get("learner", {}).get("target", "")
        for known in ("ppo", "dqn"):
            if target == f"smartsom.learning.{known}.create_learner":
                method = known
        obs_target = learning[0].get("observation", {}).get("target", "")
        for package, label in (("base", "base"), ("social", "sl")):
            if (
                obs_target
                == f"smartsom.representations.{package}.observation.create_observation"
            ):
                observation = label
    control = {"base": "distributed", "central": "centralized"}.get(
        algorithm_doc.get("control"), algorithm_doc.get("control", "custom")
    )
    return method, observation, control


def _scale_from_factory(factory_doc: dict) -> tuple[str, str]:
    factory = factory_doc.get("factory", {})
    factory_id = factory.get("factory_id", "custom")
    scale = {
        "base_small": "small",
        "small_factory": "small",
        "small_target": "small-wide",
        "medium_factory": "medium",
        "base_medium": "medium",
    }.get(factory_id, "custom")
    if factory_id == "base_target":
        grid = factory.get("grid", {})
        scale = {(27, 12): "medium", (29, 19): "medium-wide"}.get(
            (grid.get("width"), grid.get("height")), "custom"
        )
    return factory_id, scale


def _format_seeds(seeds: list[int]) -> str | None:
    if not seeds:
        return None
    if len(seeds) == 1:
        return f"s{seeds[0]}"
    contiguous = all(b == a + 1 for a, b in zip(seeds, seeds[1:]))
    if contiguous and len(seeds) >= 3:
        return f"s{seeds[0]}-{seeds[-1]}"
    if len(seeds) <= 4:
        return "s" + ",".join(map(str, seeds))
    return f"s{seeds[0]}..+{len(seeds) - 1}"


def build_auto_summary(facts: dict) -> str:
    """Deterministic local one-line summary derived purely from compiled facts."""
    parts = []
    scales = facts.get("scales") or []
    factory_ids = facts.get("factory_ids") or []
    if len(scales) == 1 and scales[0] != "custom":
        parts.append(scales[0])
    elif len(factory_ids) == 1:
        parts.append(factory_ids[0])
    elif scales:
        parts.append("/".join(scales[:2]))

    algorithms = facts.get("algorithms") or []
    observations = facts.get("observations") or []
    if algorithms:
        algo_text = "/".join(algorithms[:2]) + (
            f"+{len(algorithms) - 2}" if len(algorithms) > 2 else ""
        )
        if len(observations) == 1 and observations[0] != "custom":
            algo_text = f"{algo_text}/{observations[0]}"
        parts.append(algo_text)

    tasks = facts.get("tasks") or []
    if tasks and tasks != ["train"]:
        parts.append("/".join(tasks[:2]))

    budget = facts.get("budget") or {}
    if budget.get("train_episodes") is not None:
        parts.append(f"{budget['train_episodes']}ep")
    elif budget.get("train_ticks") is not None and "train" in tasks:
        parts.append(f"{budget['train_ticks']}ticks")

    seed_info = facts.get("seeds") or {}
    seed_list = seed_info.get("selected") or seed_info.get("all") or []
    seed_text = _format_seeds(seed_list)
    if seed_text:
        parts.append(seed_text)

    return " · ".join(parts) if parts else "smartsom-run"


def extract_facts(plan, *, created_at: float | None = None) -> dict:
    """Extract objective run facts from a ReservationPlan or DirectoryPlan."""
    is_batch = hasattr(plan, "files")
    if is_batch:
        rows = list(plan.files)
        selected_ids = (
            set(plan.selection.get("tasks", []))
            if isinstance(plan.selection, dict)
            else None
        )
        selected_rows = (
            [row for row in rows if row["id"] in selected_ids]
            if selected_ids is not None
            else rows
        )
        members = [row["compiled"] for row in rows]
        selected_members = [row["compiled"] for row in selected_rows]
        operations = plan.operations or (members[0].operations if members else {})
    else:
        members = [plan]
        selected_members = [plan]
        operations = plan.operations or {}

    tasks = _unique(member.experiment.run.task for member in members)
    all_seeds = _unique(
        seed
        for member in members
        for seed in (
            member.experiment.train_seeds
            if (not is_batch and member.experiment.run.tune != "off")
            else (member.experiment.run.seed,)
        )
    )
    selected_seeds = _unique(
        seed
        for member in selected_members
        for seed in (
            member.experiment.train_seeds
            if (not is_batch and member.experiment.run.tune != "off")
            else (member.experiment.run.seed,)
        )
    )

    algorithms, observations, controls = [], [], []
    factory_ids, scales, scenarios = [], [], []
    machines, agvs, orders = [], [], []
    episodes_values, ticks_values = [], []
    pe_rounds, pe_cases, final_evals, test_cases = [], [], [], []

    for member in members:
        docs = dict(member.documents)
        algo_doc = yaml.safe_load(docs["algorithm"]) if "algorithm" in docs else {}
        factory_doc = yaml.safe_load(docs["factory"]) if "factory" in docs else {}
        method, obs, control = _policy_summary(algo_doc)
        ex = member.experiment
        if ex.run.task in {"evaluate", "smoke"}:
            for recipe in ex.run.methods:
                algorithms.append(method if recipe == "configured" else recipe)
                observations.append("base" if recipe in {"rule", "random"} else obs)
        else:
            algorithms.append(method)
            observations.append(obs)
        controls.append(control)
        factory_id, scale = _scale_from_factory(factory_doc)
        factory_ids.append(factory_id)
        scales.append(scale)
        for name, path, _ in member.source_files:
            if name in {"scenario", "factory"}:
                scenarios.append(Path(path).name)
        machines.append(member.machines)
        agvs.append(member.agvs)
        orders.append(member.orders)
        if ex.run.task == "train":
            episodes_values.append(ex.train_episodes)
            ticks_values.append(ex.train_ticks if ex.train_episodes is None else None)
        pe_rounds.append(len(ex.periodic_evaluation.milestones))
        pe_cases.append(ex.periodic_evaluation_cases)
        final_evals.append(bool(ex.final_evaluation))
        test_cases.append(ex.test_cases)

    def _single_or_none(values):
        uniq = _unique(values)
        return uniq[0] if len(uniq) == 1 else None

    facts = {
        "host": host_identity(),
        "created_at": created_at,
        "input_type": "batch" if is_batch else "single",
        "task": tasks[0] if len(tasks) == 1 else "mixed",
        "tasks": tasks,
        "task_count": len(members),
        "selected_task_count": len(selected_members),
        "seeds": {
            "selected": selected_seeds,
            "all": all_seeds,
        },
        "budget": {
            "train_episodes": _single_or_none(episodes_values),
            "train_ticks": _single_or_none(ticks_values),
            "periodic_evaluation_rounds": _single_or_none(pe_rounds),
            "periodic_evaluation_cases": _single_or_none(pe_cases),
            "final_evaluation": _single_or_none(final_evals),
            "test_cases": _single_or_none(test_cases),
        },
        "algorithms": _unique(algorithms),
        "observations": _unique(observations),
        "controls": _unique(controls),
        "scenarios": _unique(scenarios),
        "factory_ids": _unique(factory_ids),
        "scales": _unique(scales),
        "machines": _unique(machines),
        "agvs": _unique(agvs),
        "orders": _unique(orders),
        "concurrency": {
            "mode": operations.get("concurrency", {}).get("mode"),
            "max_concurrent": operations.get("concurrency", {}).get("max_concurrent"),
            "evaluation_processes": operations.get("concurrency", {}).get(
                "evaluation_processes"
            ),
        },
        "tracking": {
            "mode": operations.get("tracking", {}).get("mode"),
            "project": operations.get("tracking", {}).get("project"),
            "tensorboard": operations.get("tracking", {}).get("tensorboard"),
        },
    }
    return facts


def requires_meta(plan) -> bool:
    """Formal train/evaluate runs require short_title and short_description."""
    if hasattr(plan, "files"):
        selected = set(plan.selection["tasks"]) if plan.selection else None
        return any(
            row["compiled"].experiment.run.task in REQUIRED_FORMAL_TASKS
            for row in plan.files
            if selected is None or row["id"] in selected
        )
    return plan.experiment.run.task in REQUIRED_FORMAL_TASKS


def missing_required_fields(meta: RunMeta | dict | None) -> list[str]:
    parsed = coerce_meta(meta)
    missing = []
    if not parsed.short_title:
        missing.append("short_title")
    if not parsed.short_description:
        missing.append("short_description")
    return missing


def prompt_missing_meta(
    meta: RunMeta | dict | None,
    facts: dict,
    *,
    input_fn=None,
    output=None,
) -> RunMeta:
    """Interactively collect missing required presentation metadata in a TTY."""
    stream = sys.stderr if output is None else output
    reader = input if input_fn is None else input_fn
    current = coerce_meta(meta)
    missing = missing_required_fields(current)
    if not missing:
        return current

    seed_info = facts.get("seeds") or {}
    seeds = seed_info.get("selected") or seed_info.get("all") or []
    summary = build_auto_summary(facts)
    concurrency = facts.get("concurrency") or {}
    tracking = facts.get("tracking") or {}
    print(
        f"\n[SmartSOM 启动确认] {summary}\n"
        f"  任务数: {facts.get('selected_task_count', facts.get('task_count'))} · "
        f"Seeds: {seeds} · "
        f"并发: {concurrency.get('max_concurrent')} ({concurrency.get('mode')}) · "
        f"W&B: {tracking.get('mode')}",
        file=stream,
        flush=True,
    )

    updates = {}
    prompts = {
        "short_title": f"短标题 short_title（如 a5，最多 {SHORT_TITLE_LIMIT} 字）: ",
        "short_description": f"简短说明 short_description（如 K4 五车，全角色学习，最多 {SHORT_DESCRIPTION_LIMIT} 字）: ",
    }
    for field in missing:
        while True:
            print(prompts[field], end="", file=stream, flush=True)
            try:
                raw = reader()
            except (EOFError, KeyboardInterrupt) as exc:
                raise ConfigurationError(
                    "launch cancelled while waiting for run metadata"
                ) from exc
            try:
                candidate = merge_meta(current, {**updates, field: raw})
                updates[field] = getattr(candidate, field)
                break
            except Exception as exc:
                print(f"  输入无效：{exc}，请重新输入。", file=stream, flush=True)
    return merge_meta(current, updates)
