"""Public evaluation contract backed by the shared grid execution workflow."""

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from smartsom.config.codec import ConfigurationError, canonical_json, primitive


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_dir: Path
    status: str
    completed: int
    failed: int
    engineering_failures: int
    results: tuple[dict, ...]
    checkpoint: Path


def _summary(rows, requested):
    completed = sum(row["status"] == "completed" for row in rows)
    failed = len(rows) - completed
    engineering = sum(row.get("engineering_failure", False) for row in rows)
    aggregates = []
    for case_id, algorithm_id in sorted(
        {(row["case_id"], row["algorithm_id"]) for row in rows}
    ):
        group = [
            row
            for row in rows
            if (row["case_id"], row["algorithm_id"]) == (case_id, algorithm_id)
        ]
        values = [row["makespan"] for row in group if row["status"] == "completed"]
        aggregates.append(
            {
                "case_id": case_id,
                "algorithm_id": algorithm_id,
                "completed": len(values),
                "failed": len(group) - len(values),
                "makespan_mean": mean(values) if values else None,
                "makespan_sample_std": stdev(values) if len(values) > 1 else None,
            }
        )
    return {
        "completed": completed,
        "failed": failed,
        "engineering_failures": engineering,
        "pending": requested - len(rows),
        "aggregates": aggregates,
    }


def evaluate_checkpoint(source, options, *, output_root=None) -> EvaluationResult:
    """The established evaluation entry delegates to the shared grid workflow."""
    from smartsom.config.experiment import EvaluationOptions
    from smartsom.experiments.production_evaluation import evaluate

    values = (
        primitive(options)
        if hasattr(options, "model_dump") or hasattr(options, "__dataclass_fields__")
        else vars(options)
    )
    try:
        frozen = EvaluationOptions.model_validate_json(canonical_json(values))
    except ValueError as exc:
        raise ConfigurationError(f"invalid evaluation options: {exc}") from exc
    return evaluate(source, frozen, output_root=output_root)
