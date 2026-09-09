"""Construct and close the selected real training backend, with no PPO updates."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from smartsom.config.codec import primitive
from smartsom.config.training import ResolvedTrainingRun
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.training_lifecycle import TrainingLifecycle
from smartsom.learning.checkpoint import require_backend
from smartsom.learning.training_extensions import (
    bind_training_extensions,
    environment_arguments,
)


def probe_training_backend(resolved, controls, *, output_root=None):
    if not isinstance(resolved, ResolvedTrainingRun):
        raise TypeError("backend probe requires ResolvedTrainingRun")
    resolved = bind_training_extensions(resolved)
    spec = resolved.algorithm.algorithm
    dependencies = require_backend(spec.provider)
    lifecycle = TrainingLifecycle(resolved, controls)
    from smartsom.learning.training_state import isolated_rng

    with isolated_rng():
        if spec.provider == "rllib.resource_ppo":
            from smartsom.learning.pettingzoo import SmartSOMParallelEnv
            from smartsom.learning.rllib_resource import train

            env = SmartSOMParallelEnv(
                resolved.episode(0).input,
                spec.projection,
                limits=resolved.run.budget.limits(),
                **environment_arguments(resolved),
            )
        else:
            from smartsom.learning.gymnasium import SchedulingEnv

            if spec.provider == "rllib.ppo":
                from smartsom.learning.rllib import train
            else:
                from smartsom.learning.sb3 import train
            env = SchedulingEnv(
                resolved.episode(0).input,
                spec.projection,
                limits=resolved.run.budget.limits(),
                observation_kind="masked" if spec.provider == "rllib.ppo" else "plain",
                strict_actions=True,
                **environment_arguments(resolved),
            )
        root = (
            Path(output_root)
            if output_root is not None
            else Path(resolved.run.output_root) / ".probes"
        )
        run_dir = root.resolve() / f"probe-{uuid4().hex}"
        run_dir.mkdir(parents=True)
        report = {
            "provider": spec.provider,
            "run_dir": str(run_dir),
            "source": source_identity(),
            "dependencies": dependencies,
            "configured_budget": primitive(resolved.run.budget),
        }
        try:
            result = train(
                resolved,
                env,
                SimpleNamespace(run_dir=run_dir, probe_only=True),
                run_dir / "unused-checkpoint",
                lifecycle=lifecycle,
            )
            report.update(status="passed", **result)
            write_json(run_dir / "probe.json", report)
            return report
        except BaseException as exc:
            write_json(
                run_dir / "probe.json",
                {
                    **report,
                    "status": "failed",
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                },
            )
            raise
        finally:
            env.close()
