"""Public Searcher delegation for scientifically censored evaluation results."""

from ray.tune.search import Searcher


class CensoredEvaluationSearch(Searcher):
    """Keep Tune success separate from a candidate with no complete objectives.

    Ray 2.58 forwards a list containing None to Optuna, leaving that Optuna trial
    RUNNING. Its public no-result completion path instead records PRUNED. This
    adapter changes only that notification, not proposals, scores or scheduling.
    Normal budget exhaustion before the first panel likewise has no objective.
    """

    def __init__(self, delegate):
        super().__init__(metric=delegate.metric, mode=delegate.mode)
        self.delegate = delegate

    def suggest(self, trial_id):
        return self.delegate.suggest(trial_id)

    def set_search_properties(self, metric, mode, config, **spec):
        return self.delegate.set_search_properties(metric, mode, config, **spec)

    def on_trial_result(self, trial_id, result):
        return self.delegate.on_trial_result(trial_id, result)

    def on_trial_complete(self, trial_id, result=None, error=False):
        if error:
            result = None
        elif result and any(result.get(key) is None for key in self.metric):
            if result.get("search_evaluation/status") not in {
                "censored",
                "not_evaluated",
            }:
                raise ValueError("completed search trial has no evaluated objectives")
            result = None
        return self.delegate.on_trial_complete(trial_id, result=result, error=error)

    def save(self, checkpoint_path):
        self.delegate.save(checkpoint_path)

    def restore(self, checkpoint_path):
        self.delegate.restore(checkpoint_path)
