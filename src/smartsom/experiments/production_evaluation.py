"""Paired grid evaluation using the established evaluation result contract."""

import json
from contextvars import copy_context
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.experiment import EvaluationOptions, ExperimentConfig
from smartsom.config.production import (
    AlgorithmConfig,
    ProductionRecipe,
    prepare_experiment,
)
from smartsom.config.study import study_roots
from smartsom.config.training import episode_root
from smartsom.experiments.evaluation import EvaluationResult, _summary
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.packaging import import_bundle, model_locator
from smartsom.experiments.references import protect_model_reference
from smartsom.learning.checkpoint import file_hash
from smartsom.telemetry.runtime import backend_diagnostics, bind, emit, operation


@operation("evaluation")
def evaluate(source, options, *, output_root=None):
    from smartsom.experiments.production import run
    from smartsom.learning.production import LearnedProductionDriver

    options = EvaluationOptions.model_validate_json(json.dumps(primitive(options)))
    if options.render_replication > options.replications:
        raise ValueError("render_replication exceeds evaluation replications")
    directory = Path(output_root or "runs").resolve() / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-evaluation-" + uuid4().hex[:10]
    )
    directory.mkdir(parents=True)
    bind(directory)
    record = {
        "schema": "smartsom.evaluation/v1",
        "id": directory.name,
        "kind": "evaluation",
        "status": "preparing",
        "stage": "checkpoint_selection",
        "source": source_identity(),
        "options": primitive(options),
        "results": [],
        "cases": [],
        "requested": 0,
        "checkpoint": None,
        "paths": {"runs": "evidence/runs", "summary": "summary.json"},
    }

    def persist():
        record.update(_summary(record["results"], record["requested"]))
        write_json(directory / "run.json", record)
        write_json(
            directory / "summary.json",
            {
                "status": record["status"],
                "stage": record["stage"],
                "input_coverage": record.get("input_coverage"),
                **_summary(record["results"], record["requested"]),
            },
        )
        emit(
            "evaluation",
            {
                "stage": record["stage"],
                "evaluation_completed": sum(
                    r.get("status") == "completed" for r in record["results"]
                ),
                "evaluation_requested": record["requested"],
                "evaluation_finished": len(record["results"]),
                "status": record["status"],
            },
            total=record["requested"],
            unit="evaluation episodes",
            final=record["stage"] == "finished",
        )

    try:
        persist()
        source = Path(source).resolve()
        record["model_input"] = {
            "path": str(source),
            "sha256": file_hash(source) if source.is_file() else None,
        }
        if source.suffix.lower() == ".zip":
            record["stage"] = "model_import"
            persist()
            try:
                source = import_bundle(source, directory / "evidence/imported-model")
            except Exception as exc:
                raise ConfigurationError(f"cannot import model bundle: {exc}") from exc
            record["paths"]["imported_model"] = "evidence/imported-model"
        record["stage"] = "checkpoint_selection"
        persist()
        try:
            checkpoint = model_locator(source, options.checkpoint)
            metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        except (ValueError, OSError) as exc:
            raise ConfigurationError(
                f"cannot read {options.checkpoint} checkpoint: {exc}"
            ) from exc
        if metadata.get("schema") != "smartsom.production-checkpoint/v1":
            raise ValueError(
                "checkpoint uses incompatible actions/observations; retrain with the grid core"
            )
        recipe_path = checkpoint / "recipe.json"
        record.update(
            stage="input_preparation",
            checkpoint={
                "path": str(checkpoint),
                "provider": metadata["provider"],
                "manifest_sha256": file_hash(checkpoint / "checkpoint.json"),
            },
        )
        persist()
        if recipe_path.is_file():
            payload = json.loads(recipe_path.read_text())
            recipe = ProductionRecipe(**payload["recipe"])
            experiment = ExperimentConfig.model_validate_json(
                json.dumps(payload["config"])
            )
        elif options.scenarios:
            first_case = Path(options.scenarios[0]).expanduser().resolve()
            if first_case.is_dir():
                first_case /= "scenario.yaml"
            experiment = ExperimentConfig.model_validate_json(
                json.dumps(
                    {
                        "scenario": str(first_case),
                        "algorithm": {"source": str(checkpoint / "checkpoint.json")},
                        "seed": options.seed,
                        "validation": {"enabled": False},
                    }
                )
            )
            recipe = prepare_experiment(
                experiment,
                training=False,
                frozen_algorithm=AlgorithmConfig.model_validate_json(
                    json.dumps(metadata["algorithm"])
                ),
            ).resolved
        else:
            raise ValueError(
                "checkpoint has no frozen experiment recipe; supply explicit evaluation scenarios"
            )
        cases = [("training", recipe)]
        if options.scenarios:
            cases = []
            for path in options.scenarios:
                path = Path(path).expanduser().resolve()
                if path.is_dir():
                    path /= "scenario.yaml"
                candidate = experiment.model_copy(deep=True)
                candidate.scenario = str(path)
                candidate.seed = options.seed
                candidate.validation.enabled = False
                prepared = prepare_experiment(
                    candidate, training=False, frozen_algorithm=recipe.algorithm
                )
                cases.append((Path(path).stem, prepared.resolved))
        if len({key for key, _ in cases}) != len(cases):
            raise ValueError("evaluation scenario filenames must have distinct stems")
        selected_case = options.render_case or cases[0][0]
        if selected_case not in {key for key, _ in cases}:
            raise ValueError("render_case must name an evaluation scenario")
        policies = [("model", recipe.algorithm, checkpoint)]
        for name in options.baselines:
            if name in ("cp", "cp_sat", "pyjobshop.cp_sat"):
                raise ConfigurationError("CP-SAT has no grid production adapter")
            provider = name if name.startswith("builtin.") else "builtin." + name
            if provider not in (
                "builtin.spt",
                "builtin.first_feasible",
                "builtin.random",
                "builtin.greedy",
                "builtin.coordinated",
            ):
                baseline = Path(name.removeprefix("checkpoint:")).resolve()
                if baseline.suffix.lower() == ".zip":
                    baseline = import_bundle(
                        baseline, directory / f"evidence/baseline-{len(policies)}"
                    )
                other = model_locator(baseline)
                metadata = json.loads((other / "checkpoint.json").read_text())
                if metadata.get("schema") != "smartsom.production-checkpoint/v1":
                    raise ValueError(
                        "baseline checkpoint uses incompatible actions/observations; retrain with the grid core"
                    )
                other_algorithm = AlgorithmConfig.model_validate_json(
                    json.dumps(metadata["algorithm"])
                )
                identifier = f"checkpoint-{file_hash(other / 'checkpoint.json')[:16]}"
                policies.append((identifier, other_algorithm, other))
                continue
            policies.append((provider, AlgorithmConfig(provider=provider), None))
        if len({name for name, _, _ in policies}) != len(policies):
            raise ValueError("duplicate evaluation baseline")
        checkpoint_identities = {
            model: {
                "path": str(model),
                "provider": algorithm.provider,
                "manifest_sha256": file_hash(model / "checkpoint.json"),
                "training_snapshot": str(model / "recipe.json")
                if (model / "recipe.json").is_file()
                else None,
                "training_snapshot_sha256": file_hash(model / "recipe.json")
                if (model / "recipe.json").is_file()
                else None,
            }
            for _, algorithm, model in policies
            if model
        }
        checkpoint_identity = checkpoint_identities[checkpoint]
        entries, materialized = [], {}
        for case_index, (case_id, case) in enumerate(cases):
            for replication in range(options.replications):
                seed = episode_root(
                    options.seed, case_index * options.replications + replication
                )
                materialized[case_id, replication] = case.episode(seed)
                world_hash = digest(materialized[case_id, replication])
                for algorithm_id, algorithm, model in policies:
                    _, algorithm_seed = study_roots(
                        options.seed, case_id, replication, algorithm_id
                    )
                    entries.append(
                        {
                            "case_id": case_id,
                            "replication": replication,
                            "algorithm_id": algorithm_id,
                            "provider": algorithm.provider,
                            "world_seed": seed,
                            "world_sha256": world_hash,
                            "algorithm_seed": algorithm_seed,
                            "checkpoint": checkpoint_identities.get(model),
                        }
                    )
        write_json(directory / "plan.json", {"entries": entries})
        from smartsom.experiments.coverage import grid_input_coverage

        record["input_coverage"] = grid_input_coverage(
            recipe_path if recipe_path.is_file() else None, entries, materialized
        )
        for model in checkpoint_identities:
            protect_model_reference(model, directory / "run.json")
        record["paths"]["plan"] = "plan.json"
        record.update(
            status="running",
            stage="evaluation",
            checkpoint=checkpoint_identity,
            requested=len(cases) * options.replications * len(policies),
            cases=[{"case_id": key} for key, _ in cases],
        )
        persist()

        def evaluate_entries(controls=None):
            for case_index, (case_id, case) in enumerate(cases):
                for replication in range(options.replications):
                    seed = episode_root(
                        options.seed, case_index * options.replications + replication
                    )
                    scenario = materialized[case_id, replication]
                    for policy_index, (algorithm_id, algorithm, model) in enumerate(
                        policies
                    ):
                        show = (
                            options.render_mode
                            if policy_index == 0
                            and case_id == selected_case
                            and replication + 1 == options.render_replication
                            else None
                        )
                        context = {
                            "case": case_id,
                            "seed": seed,
                            "replication": replication + 1,
                            "algorithm": algorithm_id,
                        }
                        from contextlib import ExitStack

                        result_row = {
                            **entries[len(record["results"])],
                            "run_dir": None,
                            "makespan": None,
                            "engineering_failure": False,
                        }
                        child = None
                        try:
                            with backend_diagnostics(), ExitStack() as cleanup:
                                driver = (
                                    LearnedProductionDriver(
                                        model,
                                        scenario,
                                        deterministic=options.deterministic,
                                        seed=result_row["algorithm_seed"],
                                    )
                                    if model
                                    else None
                                )
                                if driver:
                                    cleanup.callback(driver.env.close)
                                child = run(
                                    scenario,
                                    algorithm,
                                    output_root=directory / "evidence/runs",
                                    name=f"{case_id}-{replication + 1}",
                                    policy=driver,
                                    controls=controls if show else None,
                                    verbose=options.verbose,
                                    record=options.record,
                                    full_replay=options.full_replay,
                                    context=context,
                                    input_metadata={
                                        "workload_authoring": json.loads(
                                            case.workload_source_json
                                            or case.workload_json
                                        ),
                                        "scenario_authoring": json.loads(
                                            case.settings_json
                                        ),
                                    },
                                    checkpoint_identity=checkpoint_identities.get(
                                        model
                                    ),
                                    policy_seed=result_row["algorithm_seed"],
                                )
                            manifest = json.loads((child / "run.json").read_text())
                            state = manifest["result"]
                            result_row.update(
                                status=manifest["status"],
                                reason=manifest.get("reason"),
                                run_dir=str(child.relative_to(directory)),
                                makespan=state["tick"]
                                if manifest["status"] == "completed"
                                else None,
                                return_value=state["return"],
                                replay=manifest["audit"],
                                passing_rate=len(state["completed"])
                                / max(1, len(scenario.demands)),
                            )
                            result_row["return"] = result_row.pop("return_value")
                        except BaseException as exc:
                            failed_directory = getattr(exc, "run_dir", child)
                            result_row.update(
                                status="interrupted"
                                if isinstance(exc, KeyboardInterrupt)
                                else "failed",
                                reason=type(exc).__name__,
                                error=str(exc),
                                engineering_failure=not isinstance(
                                    exc, KeyboardInterrupt
                                ),
                                run_dir=str(
                                    Path(failed_directory).relative_to(directory)
                                )
                                if failed_directory
                                else None,
                                replay={
                                    "status": "interrupted"
                                    if options.full_replay
                                    and isinstance(exc, KeyboardInterrupt)
                                    else "unavailable_no_run_evidence"
                                    if options.full_replay and failed_directory is None
                                    else "failed"
                                    if options.full_replay
                                    else "not_requested"
                                },
                            )
                            if not isinstance(exc, Exception):
                                record["results"].append(result_row)
                                persist()
                                raise
                        record["results"].append(result_row)
                        persist()
                        if result_row["status"] == "interrupted":
                            record["status"] = "interrupted"
                            persist()
                            return EvaluationResult(
                                directory,
                                "interrupted",
                                record["completed"],
                                record["failed"],
                                record["engineering_failures"],
                                tuple(record["results"]),
                                checkpoint,
                            )
            record["status"] = (
                "failed"
                if record["engineering_failures"]
                else "completed_with_failures"
                if record["failed"]
                else "completed"
            )
            record["stage"] = "finished"
            persist()
            return EvaluationResult(
                directory,
                record["status"],
                record["completed"],
                record["failed"],
                record["engineering_failures"],
                tuple(record["results"]),
                checkpoint,
            )

        if options.render_mode is None:
            return evaluate_entries()
        import threading

        from smartsom.experiments.production import RunControls
        from smartsom.studio.playback import live_window

        if threading.current_thread() is not threading.main_thread():
            raise ValueError("human rendering must be launched from the main thread")
        controls = RunControls()
        case_index = next(i for i, (key, _) in enumerate(cases) if key == selected_case)
        controls.context = {
            "case": selected_case,
            "seed": episode_root(
                options.seed,
                case_index * options.replications + options.render_replication - 1,
            ),
            "replication": options.render_replication,
        }
        results, errors = [], []

        def worker():
            try:
                results.append(evaluate_entries(controls))
            except BaseException as exc:
                errors.append(exc)
                controls.error = exc
            finally:
                controls.finished = True

        copy = copy_context()
        thread = threading.Thread(
            target=lambda: copy.run(worker), name="smartsom-evaluation"
        )
        live_window(cases[case_index][1].scenario.factory, controls, thread)
        thread.join()
        if errors:
            raise errors[0]
        return results[0]
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        exc.run_dir = directory
        try:
            persist()
        except Exception as metadata_error:
            exc.add_note(f"failure metadata could not be saved: {metadata_error}")
        raise
