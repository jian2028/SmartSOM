"""Explicit inference modes, separate selection, and same-checkpoint comparisons."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import canonical_json
from smartsom.experiments.evidence import write_json


def action_modes(options):
    return options.action_modes or ("argmax" if options.deterministic else "sampling",)


def for_mode(prepared, mode):
    if mode not in {"argmax", "sampling"}:
        raise ValueError("unknown evaluation action mode")
    data = json.loads(prepared.config_json)
    for phase in ("validation", "evaluation"):
        data[phase].pop("action_modes", None)
        data[phase]["deterministic"] = mode == "argmax"
    return replace(prepared, config_json=canonical_json(data))


def retain_candidate(root, checkpoint, update):
    """Keep each validation candidate once, outside the rolling training GC.

    Only inference payloads are retained. The finite validation schedule bounds
    these candidates; native continuation includes this tree and its hashes.
    """
    target = Path(root) / "checkpoints/mode-candidates" / f"update-{update:06d}"
    if target.exists():
        from smartsom.experiments.tuning_session import file_digests

        expected = {
            k: v
            for k, v in file_digests(checkpoint).items()
            if k.startswith(("groups/", "controllers/"))
        }
        actual = {k: v for k, v in file_digests(target).items() if k != "snapshot.json"}
        if actual != expected:
            raise ValueError("retained validation candidate changed")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(".building-" + uuid4().hex)
    staging.mkdir()
    for name in ("snapshot.json", "groups", "controllers"):
        source = Path(checkpoint) / name
        if source.is_dir():
            shutil.copytree(source, staging / name)
        elif source.is_file():
            shutil.copyfile(source, staging / name)
    from smartsom.experiments.checkpoint_retention import sync_tree

    sync_tree(staging)
    staging.rename(target)
    return target


def finish_modes(session, case_root):
    """Evaluate both selected models in every mode; never mix their rankings."""
    from smartsom.experiments.checkpoint_retention import configured
    from smartsom.experiments.composable import (
        checkpoint_path,
        evaluate_cases,
        evaluation_recipe,
        summarize,
    )
    from smartsom.experiments.evaluation_resume import selected_cases
    from smartsom.experiments.tuning_session import (
        _control_rows,
        _copy,
        _validate_cases,
        control_recipe,
    )

    prepared, root = session.prepared, session.root
    options = prepared.config.evaluation
    cases = selected_cases(prepared)
    selectors = (
        prepared.config.validation.action_modes
        if options.checkpoint == "best" and prepared.config.validation.action_modes
        else (options.checkpoint,)
    )
    selections, references = {}, {}
    for selector in selectors:
        chosen = session.session.mode_bests.get(selector)
        diagnostic = bool(chosen is not None and chosen.get("checkpoint") is None)
        if chosen and chosen.get("checkpoint"):
            checkpoint = root / "checkpoints" / chosen["checkpoint"]
        elif diagnostic:
            if options.no_eligible_best is None:
                raise ValueError(f"no eligible {selector} checkpoint")
            last = options.no_eligible_best.last_diagnostic
            if last is None or not last.enabled:
                selections[selector] = {"status": "no_eligible_best", "results": {}}
                continue
            checkpoint = checkpoint_path(root, "last")
        else:
            checkpoint = checkpoint_path(root, selector)
        selection = {
            "status": "last_diagnostic" if diagnostic else "selected",
            "selection_mode": selector,
            "checkpoint": str(checkpoint.relative_to(root)),
            "best_update": chosen.get("update") if chosen else None,
            "ranking_eligible": not diagnostic,
            "results": {},
        }
        for mode in action_modes(options):
            session._runtime("evaluation", action_mode=mode, selection_mode=selector)
            relative = Path("final") / ("selected-by-" + selector) / mode
            rows = evaluate_cases(
                for_mode(evaluation_recipe(prepared, checkpoint), mode),
                cases,
                directory=case_root / relative / "evidence",
                control_root=root,
                label="final evaluation",
                annotations={
                    "selection_mode": selector,
                    "checkpoint_selection": "last" if diagnostic else "best",
                    "ranking_eligible": not diagnostic,
                },
            )
            _validate_cases(rows, cases, "final " + mode)
            if case_root != root / "evaluation":
                _copy(
                    case_root / relative / "evidence",
                    root / "evaluation" / relative / "evidence",
                )
            output = root / "evaluation" / relative
            write_json(output / "results.json", rows)
            write_json(output / "summary.json", summarize(rows))
            selection["results"][mode] = str(
                (output / "results.json").relative_to(root)
            )
        selections[selector] = selection

    controls = session.record["tuning"].get("control_spec") or {}
    names = set(controls.get("names", ()))
    if options.initial_control:
        names.add("initial")
    for name in sorted(names):
        if name not in {"initial", "rule", "random"}:
            raise ValueError("unknown paired control")
        control_cases = (
            selected_cases(prepared, "initial_control") if name == "initial" else cases
        )
        recipe = (
            evaluation_recipe(
                prepared,
                checkpoint_path(
                    root, "initial" if configured(prepared) else "update-000000"
                ),
            )
            if name == "initial"
            else control_recipe(prepared, random=name == "random")
        )
        for mode in action_modes(options):
            session._runtime("evaluation", action_mode=mode, control=name)
            relative = Path("controls") / name / mode
            rows = _control_rows(
                evaluate_cases(
                    for_mode(recipe, mode),
                    control_cases,
                    directory=case_root / relative / "evidence",
                    control_root=root,
                    label=name + " control",
                ),
                name,
            )
            _validate_cases(rows, control_cases, name + " " + mode)
            if case_root != root / "evaluation":
                _copy(
                    case_root / relative / "evidence",
                    root / "evaluation" / relative / "evidence",
                )
            output = root / "evaluation" / relative
            write_json(output / "results.json", rows)
            write_json(output / "summary.json", summarize(rows))
            references[name + "/" + mode] = {
                "path": str((output / "results.json").relative_to(root))
            }
    result = {
        "schema": "smartsom.evaluation-modes/v1",
        "action_modes": list(action_modes(options)),
        "policy_seed_contract": "frozen case policy_root; named policy:<group> generators",
        "selections": selections,
        "controls": references,
    }
    write_json(root / "evaluation/modes.json", result)
    session.record["tuning"].update(
        evaluation_dir="evaluation", evaluation_modes=result, controls=references
    )
