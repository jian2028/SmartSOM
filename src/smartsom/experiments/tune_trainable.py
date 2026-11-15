"""Checkpointable scientific phases for the stock Ray Tune controller.

Setup/restore never constructs a model. The common bootstrap also handles Tune
restoring a phase checkpoint with the initial resource request.
"""

import json
from pathlib import Path

from ray import tune

from smartsom.experiments.tune_phase_resources import phase_resources_ready


class ExperimentTrainable(tune.Trainable):
    def setup(self, config):
        config = config.get("trial", config)
        self.execution_config = config
        self.root = Path(config["root"])
        if "search" in config:
            self.root = self.root / "trials" / self.trial_id
        self.state = {"phase": config["initial_phase"], "after_panel": "training"}
        self.session = None
        self.search = None
        self.immutable_digests = {}
        from uuid import uuid4

        self.attempt_id = uuid4().hex
        self.recorded_run_id = None
        self._record("actor_started")
        if "search" in config:
            from smartsom.config.search import Search

            self.search = Search.model_validate(config["search"])

    def _record(self, event, **facts):
        from smartsom.experiments.tune_provenance import record

        record(
            self.execution_config["root"],
            self.trial_id,
            event,
            attempt_id=self.attempt_id,
            **facts,
        )

    def _session(self):
        if self.session is None:
            from smartsom.experiments.reservation import arguments, load
            from smartsom.experiments.reservation_session import ReservationSession

            load(self.root)
            args, _ = arguments(self.root)
            args.numerical_threads = self.execution_config.get("numerical_threads", 1)
            self.session = ReservationSession(
                args,
                defer_evaluation=True,
                evaluation_only=self.state["phase"] != "training",
            )
        return self.session

    def _pending_panel(self, session):
        node = session.periodic_evaluation_node
        settings = session.ex.periodic_evaluation
        if node not in settings.milestones:
            return False
        prefix = "tick" if settings.unit == "physical_ticks" else "episode"
        panel = self.root / "periodic_evaluation" / f"{prefix}-{node:08d}"
        return not all(
            (panel / mode / "summary.json").is_file() for mode in settings.modes
        )

    def _report(self, result=None):
        phase = self.state["phase"]
        plan = self.root / "plan.json"
        if self.recorded_run_id is None and plan.exists():
            frozen = json.loads(plan.read_text())
            self.recorded_run_id = frozen["run_id"]
            self._record(
                "scientific_run_bound",
                run_id=self.recorded_run_id,
                scientific_sha256=frozen.get("scientific_sha256"),
            )
        if (
            phase == "complete"
            and self.search is not None
            and "search_evaluation" not in self.state
        ):
            self.state["search_evaluation"] = {
                "status": "not_evaluated",
                "reason": "budget_ended_before_first_panel",
            }
        objectives = {}
        if self.search is not None:
            from smartsom.experiments.tune_search import objective_names

            objectives = {
                key: self.state.get("search_objectives", {}).get(key)
                for key in objective_names(self.search)
            }
        return {
            **(result or {}),
            **objectives,
            **(
                {"search_evaluation": self.state["search_evaluation"]}
                if "search_evaluation" in self.state
                else {}
            ),
            "next_phase": "training" if phase == "training" else "evaluation",
            "done": phase == "complete",
            "execution_attempt": self.attempt_id,
        }

    def step(self):
        from smartsom.experiments import control

        try:
            control.boundary(self.root, fresh=True)
            return self._step()
        except control.StopRequested:
            if self.session is not None:
                if self.state["phase"] == "training":
                    self.session.restore_committed()
                self.session.close()
                self.session = None
            # This result stays unfinished. Tune checkpoints the phase before
            # the driver acknowledges the pause and sends its own SIGINT.
            return self._report({"control_boundary": True})
        except BaseException as error:
            self._record("execution_error", error_type=type(error).__name__)
            raise

    def _step(self):
        from smartsom.experiments import reservation_runtime as runtime

        phase = self.state["phase"]
        if phase == "complete":
            return self._report()
        resource_phase = "training" if phase == "training" else "evaluation"
        if not phase_resources_ready(
            self.trial_resources, self.execution_config["phase_bundles"][resource_phase]
        ):
            return self._report({"phase": "resource_negotiation"})
        if "search" in self.execution_config and not (self.root / "plan.json").exists():
            from smartsom.experiments.tune_hpo import prepare_suggestion

            self.root = prepare_suggestion(
                self.execution_config, self.config, self.trial_id
            )
        if self.execution_config["task"] != "train":
            from smartsom.experiments.reservation import arguments, load, smoke

            load(self.root)
            args, settings = arguments(self.root)
            {
                "smoke": smoke,
                "calibrate": runtime.calibrate,
                "evaluate": runtime.evaluate,
            }[settings.task](args)
            self.state["phase"] = "complete"
            return self._report(json.loads((self.root / "progress.json").read_text()))
        session = self._session()
        if phase == "training":
            if self._pending_panel(session):
                self.state["phase"] = "periodic_evaluation"
                result = session.progress()
                session.close()
                self.session = None
                return self._report(result)
            result = session.step()
            following = (
                "final_evaluation"
                if result["done"]
                and session.ex.final_evaluation
                and session.budget_done()
                else "complete"
                if result["done"]
                else "training"
            )
            if self._pending_panel(session):
                self.state.update(phase="periodic_evaluation", after_panel=following)
            else:
                self.state["phase"] = following
        elif phase == "periodic_evaluation":
            from smartsom.experiments.reservation_periodic_evaluation import (
                evaluate_periodically,
            )

            evaluate_periodically(session)
            self.state["phase"] = self.state["after_panel"]
            result = session.progress()
            if "search" in self.execution_config:
                from smartsom.experiments.tune_search import (
                    objective_names,
                    panel_objective,
                )

                prefix = (
                    "tick"
                    if session.ex.periodic_evaluation.unit == "physical_ticks"
                    else "episode"
                )
                panel = (
                    self.root
                    / "periodic_evaluation"
                    / f"{prefix}-{session.periodic_evaluation_node:08d}"
                )
                self.state["search_objectives"] = {
                    key: panel_objective(panel, objective)
                    for key, objective in zip(
                        objective_names(self.search),
                        self.search.objectives,
                        strict=True,
                    )
                }
                self.state["search_evaluation"] = {
                    "panel": str(panel.relative_to(self.root)),
                    "status": "observed"
                    if all(
                        v is not None for v in self.state["search_objectives"].values()
                    )
                    else "censored",
                }
        else:
            session.final_evaluate()
            self.state["phase"] = "complete"
            result = session.progress()
        # Equal resource layouts do not restart an actor. Its old session must
        # still be retired before switching between learner and inference state.
        if self.state["phase"] != phase and self.session is not None:
            self.session.close()
            self.session = None
        return self._report(result)

    def save_checkpoint(self, checkpoint_dir):
        from smartsom.experiments.tune_checkpoint import save
        from smartsom.experiments.tune_provenance import snapshot

        return save(
            self.root,
            checkpoint_dir,
            self.state,
            immutable_digests=self.immutable_digests,
            execution=snapshot(
                self.execution_config["root"], self.trial_id, self.attempt_id
            ),
        )

    def load_checkpoint(self, checkpoint_dir):
        from smartsom.experiments.tune_checkpoint import read_metadata, restore
        from smartsom.experiments.tune_provenance import merge, pending_events, validate

        if "search" in self.execution_config:
            # Native evidence for each suggestion is isolated from its template.
            self.root = Path(self.execution_config["root"]) / "trials" / self.trial_id
        execution = read_metadata(checkpoint_dir).get("execution")
        validate(execution, self.trial_id)
        pending_events(
            self.execution_config["root"], self.trial_id, execution["events"]
        )
        self.state = restore(checkpoint_dir, self.root, trial_id=self.trial_id)
        merge(self.execution_config["root"], self.trial_id, execution["events"])
        self._record(
            "checkpoint_restored",
            source_attempt=execution["producer_attempt"],
            source_checkpoint_digest=(
                Path(checkpoint_dir) / "phase.sha256"
            ).read_text(),
        )
        self.immutable_digests.clear()

    def cleanup(self):
        import sys

        try:
            if self.session is not None:
                self.session.close()
                self.session = None
        finally:
            # The actor owns its SDK service. The driver's control ContextVar
            # does not cross Ray processes, so its finalizer cannot join this
            # service after the last tracking run has finished.
            wandb = sys.modules.get("wandb")
            if wandb is not None:
                wandb.teardown()
