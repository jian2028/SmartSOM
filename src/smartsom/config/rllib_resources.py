"""Profile-only resource arguments for public RLlib AlgorithmConfig methods."""

from pydantic import Field, model_validator

from smartsom.config.codec import StrictModel


class Driver(StrictModel):
    num_cpus_for_main_process: float = Field(default=1, gt=0, allow_inf_nan=False)
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class Learners(StrictModel):
    num_learners: int = Field(default=0, ge=0, strict=True)
    num_cpus_per_learner: float = Field(default=1, gt=0, allow_inf_nan=False)
    num_gpus_per_learner: float = Field(default=0, ge=0, allow_inf_nan=False)
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class EnvRunners(StrictModel):
    num_env_runners: int | None = Field(default=None, ge=0, strict=True)
    num_cpus_per_env_runner: float = Field(default=1, gt=0, allow_inf_nan=False)
    num_gpus_per_env_runner: float = Field(default=0, ge=0, allow_inf_nan=False)
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class EvaluationResources(StrictModel):
    num_cpus_per_env_runner: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    num_gpus_per_env_runner: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class RLlibResources(StrictModel):
    resources: Driver = Field(default_factory=Driver)
    learners: Learners = Field(default_factory=Learners)
    env_runners: EnvRunners = Field(default_factory=EnvRunners)
    evaluation_num_env_runners: int | None = Field(default=None, gt=0, strict=True)
    evaluation_config: EvaluationResources = Field(default_factory=EvaluationResources)

    @model_validator(mode="after")
    def local_memory_owned_by_driver(self):
        if self.learners.num_learners == 0 and self.learners.memory_gib is not None:
            raise ValueError(
                "local learner memory belongs in rllib.resources.memory_gib"
            )
        if (
            self.env_runners.num_env_runners == 0
            and self.env_runners.memory_gib is not None
        ):
            raise ValueError(
                "local EnvRunner memory belongs in rllib.resources.memory_gib"
            )
        return self

    @staticmethod
    def _arguments(settings, custom_key):
        values = settings.model_dump(exclude_none=True, exclude={"memory_gib"})
        if settings.memory_gib is not None:
            values[custom_key] = {"smartsom_memory_gib": settings.memory_gib}
        return values

    def arguments(self):
        return {
            name: self._arguments(getattr(self, name), key)
            for name, key in (
                ("resources", "custom_resources_for_main_process"),
                ("learners", "custom_resources_per_learner"),
                ("env_runners", "custom_resources_per_env_runner"),
            )
        }

    def evaluation_arguments(self):
        return self._arguments(
            self.evaluation_config, "custom_resources_per_env_runner"
        )

    def panel_workers(self, concurrency):
        return self.evaluation_num_env_runners or min(
            concurrency.evaluation_processes,
            concurrency.evaluation_cpus or concurrency.evaluation_processes,
        )

    def scientific_task_bundles(self, *, evaluation_workers=0):
        """Scientific tasks reserve their coordinator and optional Ray case workers."""
        bundle = {"CPU": self.resources.num_cpus_for_main_process}
        if self.resources.memory_gib is not None:
            bundle["smartsom_memory_gib"] = self.resources.memory_gib
        evaluation = self.evaluation_config
        worker = {"CPU": evaluation.num_cpus_per_env_runner or 1}
        if evaluation_workers and evaluation.num_gpus_per_env_runner:
            raise ValueError("Rule/Random evaluation does not use GPU workers")
        if evaluation.memory_gib is not None:
            worker["smartsom_memory_gib"] = evaluation.memory_gib
        return [bundle] + [dict(worker) for _ in range(evaluation_workers)]

    def declared_bundles(self, *, algorithm, evaluation_workers=0):
        """Framework-free admission check for the supported Ray 2.60 profile.

        Runtime still validates the official Algorithm resource request. This
        arithmetic does not allocate resources or choose worker counts.
        """
        if algorithm not in {"ppo", "dqn"}:
            raise ValueError(f"unsupported RLlib algorithm: {algorithm}")
        driver, learner, runner = self.resources, self.learners, self.env_runners

        def bundle(cpu, gpu, memory):
            values = {"CPU": cpu, "GPU": gpu}
            if memory is not None:
                values["smartsom_memory_gib"] = memory
            return values

        bundles = [
            bundle(
                max(driver.num_cpus_for_main_process, learner.num_cpus_per_learner)
                if learner.num_learners == 0
                else driver.num_cpus_for_main_process,
                learner.num_gpus_per_learner if learner.num_learners == 0 else 0,
                driver.memory_gib,
            )
        ]
        bundles += [
            bundle(
                learner.num_cpus_per_learner,
                learner.num_gpus_per_learner,
                learner.memory_gib,
            )
            for _ in range(learner.num_learners)
        ]
        count = runner.num_env_runners
        if count is None:
            count = 2 if algorithm == "ppo" else 0
        bundles += [
            bundle(
                runner.num_cpus_per_env_runner,
                runner.num_gpus_per_env_runner,
                runner.memory_gib,
            )
            for _ in range(count)
        ]
        evaluation = self.evaluation_config
        bundles += [
            bundle(
                evaluation.num_cpus_per_env_runner
                if evaluation.num_cpus_per_env_runner is not None
                else runner.num_cpus_per_env_runner,
                evaluation.num_gpus_per_env_runner
                if evaluation.num_gpus_per_env_runner is not None
                else runner.num_gpus_per_env_runner,
                evaluation.memory_gib
                if evaluation.memory_gib is not None
                else runner.memory_gib,
            )
            for _ in range(evaluation_workers)
        ]
        return bundles
