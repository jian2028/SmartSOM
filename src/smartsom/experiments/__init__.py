"""Execute a resolved experiment and retain its local evidence."""

from smartsom.experiments.runner import RunFailedError, RunResult, run_one

__all__ = ["RunFailedError", "RunResult", "run_one", "BatchResult", "run_batch"]

from smartsom.experiments.batch import BatchResult, run_batch
from smartsom.experiments.training import TrainingFailedError, TrainingResult, train_one

__all__ += ["TrainingFailedError", "TrainingResult", "train_one"]
