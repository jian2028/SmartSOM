"""One Tune adapter for heterogeneous official RLlib Algorithms.

All sampling, updates, evaluation and model state remain owned by the official
Algorithm. In particular this adapter does not schedule child workers itself.
"""

import json
import time
from hashlib import sha256
from math import isfinite
from pathlib import Path
from uuid import uuid4

from ray import tune
from ray.tune.result import AUTO_RESULT_KEYS, DEBUG_METRICS

CHECKPOINT_SCORE = "smartsom_checkpoint_score"


class CheckpointReferences(tune.Callback):
    """Publish references after Tune commits its official checkpoints."""

    def __init__(self):
        self.references = {}
        self.trackers = {}

    def get_state(self):
        return self.references

    def set_state(self, state):
        self.references = state

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        from smartsom.experiments.evidence import write_json

        config = trial.config.get("trial", trial.config)
        if config.get("scientific_task"):
            # A completed calibration has a Tune checkpoint, not model best/last.
            return
        row = self.references.setdefault(trial.trial_id, {"best": None})
        current = {
            "path": checkpoint.path,
            "algorithm_iteration": trial.last_result["algorithm_iteration"],
        }
        row["last"] = current
        score = trial.last_result.get(CHECKPOINT_SCORE, float("inf"))
        if isfinite(score) and (row["best"] is None or score <= row["best"]["score"]):
            row["best"] = {**current, "score": score}
        config = trial.config.get("trial", trial.config)
        root = config.get("root")
        if root and "search" in config:
            root = str(Path(root) / "trials" / trial.trial_id)
        directory = config.get("checkpoint_reference_directory", root)
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            write_json(Path(directory) / f"checkpoint-refs-{trial.trial_id}.json", row)
        if root:
            metrics = trial.last_result
            progress = {
                key: metrics[key]
                for key in (
                    "episodes",
                    "physical_ticks",
                    "algorithm_iteration",
                    "training_iteration",
                    "status",
                    "time_total_s",
                )
                if key in metrics
            }
            progress.update(
                committed_physical_ticks=metrics.get("physical_ticks", 0),
                event_id=f"{trial.trial_id}/{metrics['training_iteration']}",
                checkpoint=checkpoint.path,
            )
            write_json(Path(root) / "progress.json", progress)
            tracker = self.trackers.get(trial.trial_id)
            if tracker is None:
                from smartsom.config.base_diagnostic import load_inputs
                from smartsom.telemetry.experiment_tracking import ExperimentTracking

                experiment = load_inputs(Path(root) / "inputs/experiment.yaml")[0]
                tracker = ExperimentTracking(
                    root, experiment.tracking, job_type="train"
                )
                tracker.start(
                    json.loads((Path(root) / "manifest.json").read_text()),
                    experiment.model_dump(mode="json", by_alias=True),
                )
                self.trackers[trial.trial_id] = tracker
            tracker.log({"phase": "train", **progress})
            panel = metrics.get("evaluation", {}).get("env_runners")
            if panel is not None:
                tracker.log(
                    {
                        "phase": "periodic_evaluation",
                        "selection": panel["selection_mode"],
                        "training_progress": panel.get("actual_progress", {}),
                        "event_id": progress["event_id"] + "/periodic_evaluation",
                        "metrics": {k: v for k, v in panel.items() if k != "cases"},
                    }
                )
            for selection, final in metrics.get("final_evaluation", {}).items():
                if not isinstance(final, dict) or "panel_split" not in final:
                    continue
                tracker.log(
                    {
                        "phase": "evaluation",
                        "selection": final["selection_mode"],
                        "checkpoint_selection": selection,
                        "checkpoint": final["checkpoint"],
                        "training_progress": {
                            key: metrics.get(key, 0)
                            for key in ("episodes", "physical_ticks")
                        },
                        "event_id": progress["event_id"] + "/final/" + selection,
                        "metrics": {
                            k: v
                            for k, v in final.items()
                            if k not in {"cases", "checkpoint"}
                        },
                    }
                )

    def on_trial_complete(self, iteration, trials, trial, **info):
        tracker = self.trackers.pop(trial.trial_id, None)
        if tracker is not None:
            tracker.close()

    def on_trial_error(self, iteration, trials, trial, **info):
        tracker = self.trackers.pop(trial.trial_id, None)
        if tracker is not None:
            tracker.close(failed=True)

    def on_experiment_end(self, trials, **info):
        for tracker in self.trackers.values():
            tracker.close()
        self.trackers.clear()


def checkpoint_config(*, keep_all=False):
    """Official Tune retains the best scored checkpoint plus the latest one."""
    return tune.CheckpointConfig(
        checkpoint_frequency=1,
        checkpoint_at_end=True,
        num_to_keep=None if keep_all else 1,
        checkpoint_score_attribute=CHECKPOINT_SCORE,
        checkpoint_score_order="min",
    )


def score_checkpoint(result):
    """A partial or absent panel cannot qualify a checkpoint as best."""
    panel = result.get("evaluation", {}).get("env_runners", {})
    value = panel.get("joint_cost")
    qualifies = (
        panel.get("complete_panel") is True
        and panel.get("panel_split") == "periodic_evaluation"
        and panel.get("eligible_for_best", True)
        and value is not None
        and isfinite(value)
    )
    return {**result, CHECKPOINT_SCORE: value if qualifies else float("inf")}


def checkpoint_references(result):
    """Return official checkpoints only, explicitly allowing no valid best."""
    candidates = [
        (checkpoint, metrics[CHECKPOINT_SCORE])
        for checkpoint, metrics in result.best_checkpoints or ()
        if isfinite(metrics.get(CHECKPOINT_SCORE, float("inf")))
    ]
    return {
        "last": result.checkpoint,
        "best": min(candidates, key=lambda item: item[1])[0] if candidates else None,
    }


class RLlibTrainable(tune.Trainable):
    @classmethod
    def default_resource_request(cls, config):
        config = config.get("trial", config)
        if config.get("scientific_task"):
            from ray.tune.execution.placement_groups import PlacementGroupFactory

            return PlacementGroupFactory(config["resource_bundles"])
        settings = config["algorithm_config"]
        return settings.algo_class.default_resource_request(settings)

    def setup(self, config):
        values = config
        config = config.get("trial", config)
        self.search = None
        if "search" in config:
            from smartsom.config.search import Search
            from smartsom.experiments.rllib_execution import trial_config
            from smartsom.experiments.tune_hpo import prepare_suggestion

            self.search = Search.model_validate(config["search"])
            config = trial_config(prepare_suggestion(config, values, self.trial_id))
        # Delay construction so restoring does not initialize then discard a
        # second Algorithm. Tune calls load_checkpoint before the first step.
        self.algorithm = None
        self.scientific_task = config.get("scientific_task", False)
        self.task = config.get("task", "train")
        self.algorithm_config = config.get("algorithm_config")
        self.execution = config.get("execution", {})
        self.root = Path(config["root"]) if "root" in config else None
        self.execution_id = uuid4().hex
        if self.root is not None:
            from smartsom.experiments.tune_provenance import record

            record(
                self.root, self.trial_id, "actor_started", attempt_id=self.execution_id
            )
        self.checkpoint_reference_directory = config.get(
            "checkpoint_reference_directory", self.root
        )
        seconds = self.execution.get("seconds")
        self.state = {
            "schema": "smartsom.rllib-runtime/v1",
            "phase": "training",
            "deadline_unix": time.time() + seconds if seconds is not None else None,
            "evaluated_milestones": [],
            "search_objectives": {},
            "search_evaluation": "not_evaluated",
            "smoke_rows": None,
        }
        if (
            self.execution.get("periodic_evaluation")
            and self.algorithm_config.evaluation_interval
        ):
            raise ValueError(
                "scientific milestones require automatic iteration evaluation disabled"
            )

    def _progress(self):
        return {
            "algorithm_initialized_unix": self.algorithm.metrics.peek(
                "smartsom_algorithm_initialized_unix", default=0.0
            ),
            **{
                name: self.algorithm.metrics.peek(
                    "smartsom_" + name + "_lifetime", default=0
                )
                for name in ("episodes", "physical_ticks")
            },
        }

    def _timed(self, name, operation):
        started = time.perf_counter()
        try:
            return operation()
        finally:
            self.phase_seconds[name] = (
                self.phase_seconds.get(name, 0.0) + time.perf_counter() - started
            )

    def _report(self, result):
        result["phase_seconds"] = dict(self.phase_seconds)
        if self.search is not None:
            from smartsom.experiments.tune_search import objective_names

            panel = result.get("evaluation", {}).get("env_runners")
            if panel is not None:
                self.state["search_evaluation"] = (
                    "complete" if panel["complete_panel"] else "censored"
                )
                for name, objective in zip(
                    objective_names(self.search), self.search.objectives
                ):
                    _, condition, metric = objective.metric.split(".")
                    if (
                        condition != "all"
                        or metric
                        not in {"makespan", "mean_tardiness", "final_passing_rate"}
                        or objective.panel != panel["selection_mode"]
                    ):
                        raise ValueError(
                            "RLlib search requires all-case objectives under the official evaluation policy"
                        )
                    self.state["search_objectives"][name] = panel[metric]
            result.update(dict.fromkeys(objective_names(self.search)))
            result.update(self.state["search_objectives"])
            result["search_evaluation"] = {"status": self.state["search_evaluation"]}
        result["algorithm_iteration"] = self.algorithm.iteration
        if "time_this_iter_s" in result:
            result["algorithm_time_this_iter_s"] = result["time_this_iter_s"]
        # Tune owns report iteration, identity and elapsed time, including
        # initialization/final evaluation. Do not inherit the inner Algorithm's
        # shorter train() timing or collide with an evaluation-only report.
        for key in set(AUTO_RESULT_KEYS) | set(DEBUG_METRICS):
            result.pop(key, None)
        return score_checkpoint(result)

    def _boundary(self):
        if self.root is not None:
            from smartsom.experiments import control

            try:
                control.boundary(self.root, fresh=True)
            except control.StopRequested:
                return True
        return False

    def _budget_done(self, result):
        return any(
            self.execution.get(name) is not None
            and result.get(name, 0) >= self.execution[name]
            for name in ("episodes", "physical_ticks")
        ) or (
            self.state["deadline_unix"] is not None
            and time.time() >= self.state["deadline_unix"]
        )

    def _advance_if_finished(self, result):
        if self._budget_done(result):
            self.state["phase"] = (
                "final_evaluation"
                if self.execution.get("final_evaluation")
                else "complete"
            )

    def _evaluate_due(self, result):
        settings = self.execution.get("periodic_evaluation")
        if not settings:
            return False
        reached = [
            node
            for node in settings["milestones"]
            if node <= result[settings["unit"]]
            and node not in self.state["evaluated_milestones"]
        ]
        if not reached:
            return False
        from smartsom.learning.rllib_evaluation import FixedPanels

        evaluator = self.algorithm.config.custom_evaluation_function
        if not isinstance(evaluator, FixedPanels):
            raise ValueError("scientific milestones require frozen evaluation panels")
        evaluator.select_periodic()
        panel = self._timed("periodic_evaluation", self.algorithm.evaluate)[
            "env_runners"
        ]
        panel["eligible_for_best"] = self.algorithm.iteration > 0 or settings.get(
            "select_initial", False
        )
        panel["scheduled_milestones"] = reached
        panel["actual_progress"] = {
            name: result[name] for name in ("episodes", "physical_ticks")
        }
        result["evaluation"] = {"env_runners": panel}
        self.state["evaluated_milestones"].extend(reached)
        return True

    def step(self):
        self.phase_seconds = {}
        if self.scientific_task:
            return self._scientific_step()
        if self.algorithm is None:
            if self.root is not None:
                from smartsom.experiments.reservation import load

                load(self.root)
            self.algorithm = self._timed(
                "initialization", self.algorithm_config.build_algo
            )
        result = self._progress()
        if self._boundary():
            return self._report({**result, "control_boundary": True, "done": False})
        if self.task == "smoke" and self.state["smoke_rows"] is None:
            from smartsom.experiments import control
            from smartsom.experiments.reservation import arguments, smoke

            try:
                self.state["smoke_rows"] = smoke(arguments(self.root)[0])
            except control.StopRequested:
                return self._report({**result, "control_boundary": True, "done": False})
        if self.state["phase"] == "training":
            self._advance_if_finished(result)
            if self.state["phase"] == "final_evaluation" and self.execution.get(
                "final_checkpoint_selection"
            ):
                return self._report({**result, "done": False, "status": "evaluating"})
            if self.state["phase"] == "training" and self._evaluate_due(result):
                return self._report({**result, "done": False, "status": "running"})
        if self.state["phase"] == "training":
            result = self._timed("training", self.algorithm.train)
            interval = self.algorithm.config.evaluation_interval
            if not interval or self.algorithm.iteration % interval:
                # Ray can emit the previous panel during the next training
                # compile. It must not score or republish changed weights.
                result.pop("evaluation", None)
            self._evaluate_due(result)
            self._advance_if_finished(result)
            if self.state["phase"] == "final_evaluation" and self.execution.get(
                "final_checkpoint_selection"
            ):
                # Let Tune commit the final training state before evaluating
                # retained checkpoints. Its callback publishes durable references.
                return self._report({**result, "done": False, "status": "evaluating"})
        if self.state["phase"] == "final_evaluation":
            if self._boundary():
                return self._report({**result, "control_boundary": True, "done": False})
            from smartsom.learning.rllib_evaluation import FixedPanels

            evaluator = self.algorithm.config.custom_evaluation_function
            if not isinstance(evaluator, FixedPanels):
                raise ValueError(
                    "final evaluation requires the frozen test panel callback"
                )
            evaluator.select_final()
            if self.execution.get("final_checkpoint_selection"):
                result["final_evaluation"] = self._timed(
                    "final_evaluation", self._evaluate_checkpoints
                )
            else:
                result["final_evaluation"] = self._timed(
                    "final_evaluation", self.algorithm.evaluate
                )["env_runners"]
            self.state["phase"] = "complete"
        result["done"] = self.state["phase"] == "complete"
        result["status"] = "completed" if result["done"] else "running"
        if self.task == "smoke" and result["done"]:
            from smartsom.experiments.evidence import write_json

            write_json(
                self.root / "summary.json",
                {
                    "kind": "bounded-smoke-not-convergence",
                    "results": self.state["smoke_rows"]
                    + [
                        {
                            "method": self.algorithm.config.env_config["algorithm"],
                            **self._progress(),
                            "algorithm_iteration": self.algorithm.iteration,
                        }
                    ],
                },
            )
        return self._report(result)

    def _scientific_step(self):
        from smartsom.experiments import control
        from smartsom.experiments.reservation import arguments, load, smoke
        from smartsom.experiments.reservation_runtime import calibrate, evaluate, freeze

        load(self.root)
        if self._boundary():
            return {
                "control_boundary": True,
                "done": False,
                CHECKPOINT_SCORE: float("inf"),
            }
        if self.state["phase"] != "complete":
            args, _ = arguments(self.root)
            if self.task in {"calibrate", "evaluate"}:
                args.control_check = lambda: control.boundary(self.root, fresh=True)
                if self.task == "evaluate":
                    bundles = self.config.get("trial", self.config)["resource_bundles"]
                    args.evaluation_workers = len(bundles) - 1
                    worker = bundles[1]
                    args.worker_resources = {
                        "num_cpus": worker["CPU"],
                        "resources": {
                            key: value for key, value in worker.items() if key != "CPU"
                        },
                    }
                try:
                    (calibrate if self.task == "calibrate" else evaluate)(args)
                except control.StopRequested:
                    return {
                        "control_boundary": True,
                        "done": False,
                        CHECKPOINT_SCORE: float("inf"),
                    }
            else:
                freeze(self.root, args._prepared_inputs[0][-1], "smoke", {})
                try:
                    smoke(args)
                except control.StopRequested:
                    return {
                        "control_boundary": True,
                        "done": False,
                        CHECKPOINT_SCORE: float("inf"),
                    }
            self.state["phase"] = "complete"
        return {"done": True, "status": "completed", CHECKPOINT_SCORE: float("inf")}

    def _evaluate_checkpoints(self):
        references = json.loads(
            (
                Path(self.checkpoint_reference_directory)
                / f"checkpoint-refs-{self.trial_id}.json"
            ).read_text()
        )
        last = references["last"]
        if last["algorithm_iteration"] != self.algorithm.iteration:
            raise ValueError(
                "final evaluation checkpoint does not match training boundary"
            )
        panels = {}
        # Restore only official learner state into the resident Algorithm. The
        # public API syncs runners; replay, counters and training state stay last.
        try:
            for selection in self.execution["final_checkpoint_selection"]:
                reference = references[selection]
                if reference is None:
                    panels[selection] = {"status": "no_eligible_checkpoint"}
                    continue
                self._restore_evaluation_weights(reference["path"])
                panels[selection] = {
                    # Tune will replace the training-boundary last checkpoint
                    # with the completed checkpoint after restoring last below.
                    # Resolve through the committed reference file, never publish
                    # a path that official retention is about to remove.
                    "checkpoint": {
                        "selection": selection,
                        "algorithm_iteration": reference["algorithm_iteration"],
                        "references": str(
                            Path(self.checkpoint_reference_directory)
                            / f"checkpoint-refs-{self.trial_id}.json"
                        ),
                    },
                    **self.algorithm.evaluate()["env_runners"],
                }
        finally:
            self._restore_evaluation_weights(last["path"])
        return panels

    def _restore_evaluation_weights(self, path):
        self.algorithm.restore_from_path(
            Path(path) / "learner_group", component="learner_group"
        )
        # Loading an older best checkpoint can move the learner's weight version
        # backwards. Normal synchronization skips older versions. The public
        # EnvRunner.set_state contract accepts an unversioned module update to
        # load that exact state, without changing any private version counters.
        weights = self.algorithm.learner_group.get_state(
            components=["learner/rl_module"], inference_only=True
        )["learner"]["rl_module"]
        for group in (
            self.algorithm.env_runner_group,
            self.algorithm.eval_env_runner_group,
        ):
            if group is not None:
                group.foreach_env_runner(
                    lambda runner: runner.set_state({"rl_module": weights}),
                    local_env_runner=True,
                )

    def save_checkpoint(self, checkpoint_dir):
        if self.scientific_task:
            return self._save_scientific_checkpoint(checkpoint_dir)
        if self.algorithm is None:
            raise RuntimeError("cannot checkpoint an Algorithm before initialization")
        started = time.perf_counter()
        path = self.algorithm.save_to_path(checkpoint_dir)
        # Operational phase/deadline only. All training state and counters stay
        # in the official Algorithm checkpoint and its MetricsLogger.
        state = dict(self.state)
        if self.root is not None:
            from smartsom.experiments.tune_provenance import snapshot

            state["execution"] = snapshot(self.root, self.trial_id, self.execution_id)
        (Path(path) / "smartsom-runtime.json").write_text(json.dumps(state))
        (Path(path) / "smartsom-timing.json").write_text(
            json.dumps(
                {
                    "checkpoint_seconds": time.perf_counter() - started,
                }
            )
        )
        return path

    def load_checkpoint(self, checkpoint_dir):
        if self.scientific_task:
            self._load_scientific_checkpoint(checkpoint_dir)
            return
        from ray.rllib.algorithms.algorithm import Algorithm

        metadata = (Path(checkpoint_dir) / "smartsom-runtime.json").read_bytes()
        state = json.loads(metadata)
        if state.get("schema") != "smartsom.rllib-runtime/v1" or state.get(
            "phase"
        ) not in {
            "training",
            "final_evaluation",
            "complete",
        }:
            raise ValueError("invalid RLlib runtime checkpoint metadata")
        deadline = state.get("deadline_unix")
        if not isinstance(state.get("evaluated_milestones"), list):
            raise ValueError("invalid RLlib runtime checkpoint milestones")
        if "deadline_unix" not in state or (
            deadline is not None
            and (type(deadline) not in (int, float) or not isfinite(deadline))
        ):
            raise ValueError("invalid RLlib runtime checkpoint deadline")
        if self.root is not None:
            from smartsom.experiments.reservation import load
            from smartsom.experiments.tune_provenance import pending_events, validate

            load(self.root)
            validate(state.get("execution"), self.trial_id)
            pending_events(self.root, self.trial_id, state["execution"]["events"])
        if self.algorithm is not None:
            self.algorithm.stop()
            self.algorithm = None
        # A corrupt/missing official checkpoint must propagate an error.
        self.algorithm = Algorithm.from_checkpoint(checkpoint_dir)
        self.state = state
        if self.root is not None:
            from smartsom.experiments.tune_provenance import merge, record

            merge(self.root, self.trial_id, state["execution"]["events"])
            record(
                self.root,
                self.trial_id,
                "checkpoint_restored",
                attempt_id=self.execution_id,
                source_attempt=state["execution"]["producer_attempt"],
                source_checkpoint_digest=sha256(metadata).hexdigest(),
                source_checkpoint_digest_basis="smartsom-runtime.json",
            )

    def _save_scientific_checkpoint(self, checkpoint_dir):
        """Scientific reports live in Tune's checkpoint, with no model store."""
        names = (
            "manifest.json",
            "calibration.json",
            "summary.json",
            "progress.json",
            "cases.jsonl",
            "research-summary.json",
        )
        files = {
            name: (self.root / name).read_text()
            for name in names
            if (self.root / name).is_file()
        }
        files.update(
            {
                str(p.relative_to(self.root)): p.read_text()
                for p in (self.root / "case-commits").rglob("*.json")
            }
        )
        from smartsom.experiments.tune_provenance import snapshot

        payload = {
            "task": self.task,
            "state": self.state,
            "files": files,
            "file_sha256": {
                name: sha256(content.encode()).hexdigest()
                for name, content in files.items()
            },
            "plan_sha256": sha256((self.root / "plan.json").read_bytes()).hexdigest(),
            "execution": snapshot(self.root, self.trial_id, self.execution_id),
        }
        Path(checkpoint_dir, "scientific-task.json").write_text(json.dumps(payload))
        return checkpoint_dir

    def _load_scientific_checkpoint(self, checkpoint_dir):
        from smartsom.experiments.reservation import load
        from smartsom.experiments.tune_provenance import (
            merge,
            pending_events,
            record,
            validate,
        )

        load(self.root)
        payload = json.loads(Path(checkpoint_dir, "scientific-task.json").read_text())
        if (
            payload["task"] != self.task
            or payload["state"].get("schema") != "smartsom.rllib-runtime/v1"
        ):
            raise ValueError("scientific task checkpoint identity mismatch")
        if (
            payload["plan_sha256"]
            != sha256((self.root / "plan.json").read_bytes()).hexdigest()
        ):
            raise ValueError("scientific checkpoint belongs to a different frozen run")
        validate(payload["execution"], self.trial_id)
        pending_events(self.root, self.trial_id, payload["execution"]["events"])
        files = payload["files"]
        if payload["file_sha256"] != {
            name: sha256(content.encode()).hexdigest()
            for name, content in files.items()
        }:
            raise ValueError("scientific checkpoint artifact checksum mismatch")
        if payload["state"]["phase"] not in {"training", "complete"}:
            raise ValueError("invalid scientific task phase")
        required = {
            "manifest.json",
            "calibration.json" if self.task == "calibrate" else "summary.json",
        }
        if self.task == "evaluate":
            required.update({"cases.jsonl", "research-summary.json"})
        if payload["state"]["phase"] == "complete" and not required <= files.keys():
            raise ValueError("scientific checkpoint is missing completed results")
        for name in files:
            path = Path(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (
                    name
                    in {
                        "manifest.json",
                        "calibration.json",
                        "summary.json",
                        "progress.json",
                        "cases.jsonl",
                        "research-summary.json",
                    }
                    or (path.parts[0] == "case-commits" and path.suffix == ".json")
                )
            ):
                raise ValueError("invalid scientific checkpoint artifact")
        for name, content in files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.state = payload["state"]
        merge(self.root, self.trial_id, payload["execution"]["events"])
        record(
            self.root,
            self.trial_id,
            "checkpoint_restored",
            attempt_id=self.execution_id,
            source_attempt=payload["execution"]["producer_attempt"],
            source_checkpoint_digest=sha256(
                Path(checkpoint_dir, "scientific-task.json").read_bytes()
            ).hexdigest(),
            source_checkpoint_digest_basis="scientific-task.json",
        )

    def cleanup(self):
        if self.algorithm is not None:
            self.algorithm.stop()
            self.algorithm = None
