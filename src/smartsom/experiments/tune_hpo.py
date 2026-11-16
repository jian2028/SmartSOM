"""Frozen Optuna searches using the same phase Trainable as fixed experiments."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from smartsom.config.codec import safe_yaml_load
from smartsom.config.reservation import compile_reservation
from smartsom.config.search import Search, load_search, validate_paths
from smartsom.experiments.evidence import write_json
from smartsom.telemetry.runtime import bind, operation


@dataclass(frozen=True)
class SearchPlan:
    template: object
    search: Search
    calibration: dict | None = None
    calibration_inputs: bytes | None = None

    @property
    def experiment(self):
        return self.template.experiment

    @property
    def operations(self):
        return self.template.operations

    def summary(self):
        return {
            **self.template.summary(),
            "search": self.search.model_dump(mode="json"),
            "shared_calibration": self.calibration,
        }


def _shared_calibration(template):
    """Validate the original scientific binding before deriving a common limit."""
    reference = template.experiment.run.calibration
    if not reference:
        return None, None
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import calibration_identity_payload

    raw = Path(reference).read_bytes()
    report = json.loads(raw)
    with TemporaryDirectory(prefix="smartsom-search-calibration-") as folder:
        experiment = template.materialize(Path(folder))
        identity_inputs = calibration_identity_payload(load_inputs(experiment)[-1])
    identity = hashlib.sha256(identity_inputs).hexdigest()
    if (
        not isinstance(report, dict)
        or report.get("status") != "frozen"
        or report.get("calibration_identity") != identity
    ):
        raise ValueError("search requires matching completed template calibration")
    drain = report.get("drain_ticks")
    if type(drain) is not int or drain <= 0:
        raise ValueError("search calibration requires a positive integer drain_ticks")
    metadata = {
        "schema": "smartsom.search-calibration/v1",
        "policy": "shared_template_drain",
        "source_path": str(reference),
        "report": "inputs/calibration.json",
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "template_calibration_identity": identity,
        "identity_inputs": "inputs/calibration-identity.json",
        "identity_inputs_sha256": identity,
        "template_scientific_sha256": template.scientific_sha256,
        "drain_ticks": drain,
    }
    return metadata, identity_inputs


def _read_shared_calibration(root, metadata):
    """Verify the derivation using only the owning template or trial's files."""
    report = (root / "inputs/calibration.json").read_bytes()
    inputs = (root / "inputs/calibration-identity.json").read_bytes()
    identity = hashlib.sha256(inputs).hexdigest()
    parsed = json.loads(report)
    if (
        hashlib.sha256(report).hexdigest() != metadata["report_sha256"]
        or identity != metadata["identity_inputs_sha256"]
        or identity != metadata["template_calibration_identity"]
        or parsed["calibration_identity"] != identity
        or parsed["status"] != "frozen"
        or parsed["drain_ticks"] != metadata["drain_ticks"]
    ):
        raise ValueError("search calibration report identity mismatch")
    return report, inputs


def _bind_files(root, names):
    frozen = json.loads((root / "plan.json").read_text())
    for name in names:
        frozen["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(root / "plan.json", frozen)


def _freeze_search(root, plan):
    names = ["search.json"]
    if plan.calibration is not None:
        raw = (root / "inputs/calibration.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != plan.calibration["report_sha256"]:
            raise ValueError("search calibration changed after validation")
        (root / "inputs/calibration-identity.json").write_bytes(plan.calibration_inputs)
        _read_shared_calibration(root, plan.calibration)
        write_json(root / "search-calibration.json", plan.calibration)
        names.extend(["search-calibration.json", "inputs/calibration-identity.json"])
    write_json(root / "search.json", plan.search.model_dump(mode="json"))
    _bind_files(root, names)


def compile_search(path, search, **overrides):
    spec = load_search(search)
    if overrides.get("seed") not in (None, spec.train_seed):
        raise ValueError("--seed conflicts with the search's explicit train_seed")
    if overrides.get("task") not in (None, "train"):
        raise ValueError("search requires a learning train task")
    overrides.update(task="train", seed=spec.train_seed)
    template = compile_reservation(path, **overrides)
    validate_paths(spec, safe_yaml_load(dict(template.documents)["algorithm"]))
    from smartsom.telemetry.evaluation_summary import METRICS

    for objective in spec.objectives:
        if objective.metric.rsplit(".", 1)[-1] not in METRICS:
            raise ValueError(f"unknown periodic evaluation metric: {objective.metric}")
    panel = template.experiment.periodic_evaluation
    if not panel.milestones or any(
        objective.panel not in panel.modes for objective in spec.objectives
    ):
        raise ValueError(
            "search objective requires configured periodic evaluation milestones and mode"
        )
    calibration, identity_inputs = _shared_calibration(template)
    return SearchPlan(template, spec, calibration, identity_inputs)


@operation("search")
def run(plan):
    from smartsom.config.rllib_compatibility import validate_plan
    from smartsom.experiments.reservation import allocate

    validate_plan(plan)
    root = allocate(plan.template)
    _freeze_search(root, plan)
    if plan.operations["launch"]["mode"] == "background":
        from smartsom.experiments.background import launch

        return launch(root)
    return execute(root)


def execute(root, *, retry_failed=False):
    from smartsom.config.search import reject_failed_retry

    reject_failed_retry(retry_failed)
    from smartsom.experiments import control
    from smartsom.experiments.reservation import arguments, load
    from smartsom.experiments.tune_execution import fit_cohort

    root = Path(root).resolve()
    load(root)
    bind(root)
    spec = Search.model_validate_json((root / "search.json").read_text())
    _, settings = arguments(root)
    operations = json.loads((root / "execution.json").read_text())["settings"]
    storage = root / "ray/search"
    results = fit_cohort(
        [root],
        storage=storage,
        resources=operations["resources"],
        max_concurrent=operations["concurrency"]["max_concurrent"],
        max_failures=settings.max_failures,
        retry_failed=retry_failed,
        search=spec,
    )
    complete = len(results) == spec.num_samples and all(
        result.metrics.get("done") for result in results
    )
    status = (
        "failed"
        if results.errors
        else "completed"
        if complete
        else "paused"
        if control.pause_requested(root)
        else "stopped"
    )
    from smartsom.experiments.tune_batch import public_metrics
    from smartsom.experiments.tune_search import pareto_candidates

    report = {
        "directory": str(root),
        "status": status,
        "objectives": [o.model_dump(mode="json") for o in spec.objectives],
        "pareto_candidates": pareto_candidates(results, spec),
        "trials": [
            {
                "path": str(result.path),
                "metrics": public_metrics(result.metrics),
                "error": str(result.error) if result.error else None,
            }
            for result in results
        ],
    }
    write_json(root / "batch.json", report)
    write_json(
        root / "progress.json",
        {
            "status": status,
            "completed": sum(bool(r.metrics.get("done")) for r in results),
            "total": spec.num_samples,
        },
    )
    return report


def prepare_suggestion(config, values, trial_id):
    from smartsom.experiments.reservation import allocate, load

    template = Path(config["root"])
    frozen = load(template)
    root = template / "trials" / trial_id
    if (root / "plan.json").is_file():
        load(root)
        return root
    spec = Search.model_validate(config["search"])
    operations = json.loads((template / "execution.json").read_text())["settings"]
    overrides = [f"{path}={json.dumps(values[path])}" for path in spec.parameters]
    shared = None
    shared_report = None
    shared_inputs = None
    if (template / "inputs/calibration.json").is_file():
        if "search-calibration.json" not in frozen["files"]:
            raise ValueError(
                "search calibration provenance missing; create a new search"
            )
        shared = json.loads((template / "search-calibration.json").read_text())
        shared_report, shared_inputs = _read_shared_calibration(template, shared)
        overrides.extend(
            [
                "experiment.run.calibration=null",
                "experiment.drain_frozen=true",
                f"experiment.drain_ticks={shared['drain_ticks']}",
            ]
        )
    compiled = compile_reservation(
        template / "inputs/experiment.yaml",
        task="train",
        seed=spec.train_seed,
        sets=overrides,
        operations=operations,
    )
    control = template / "control"
    control.mkdir(exist_ok=True)
    # An interrupted preparation must never expose a valid plan without its
    # derivation evidence. Staging is owned by this call and excluded from trials.
    with TemporaryDirectory(prefix="suggestion-", dir=control) as folder:
        staged = Path(folder) / "candidate"
        allocate(compiled, root=staged, announce=False)
        names = ["suggestion.json"]
        if shared is not None:
            # The report calibrated the template, not the changed algorithm.
            (staged / "inputs/calibration.json").write_bytes(shared_report)
            (staged / "inputs/calibration-identity.json").write_bytes(shared_inputs)
            write_json(staged / "inputs/search-calibration.json", shared)
            names.extend(
                [
                    "inputs/calibration.json",
                    "inputs/calibration-identity.json",
                    "inputs/search-calibration.json",
                ]
            )
        write_json(
            staged / "suggestion.json",
            {
                "parameters": {path: values[path] for path in spec.parameters},
                "search": str(template),
                "trial_id": trial_id,
                "scientific_sha256": compiled.scientific_sha256,
            },
        )
        _bind_files(staged, names)
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            root.rmdir()  # Only an empty bootstrap-restored directory is accepted.
        staged.rename(root)
    return root
