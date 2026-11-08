"""Tail accounting and a separately identified optional task-window objective."""

from smartsom.domain.training import subject_rewards
from smartsom.learning.dqn.learner import BaseDQN
from smartsom.learning.dqn.replay import PackedRows
from smartsom.learning.subjects import subject_key


def tail_accounting(records, rewards, *, terminal, split_stages):
    """Count unresolved subjects, without summing duplicated team reward as loss."""
    last = {}
    for row in records:
        request = row["request"]
        last[subject_key(request, split_stages)] = request
    following = {
        subject_key(r, split_stages)
        for r in getattr(rewards, "final_requests", {}).values()
    }
    rows = []
    for key, request in last.items():
        if key in following:
            continue
        stream = subject_rewards(rewards, request)
        rows.append(
            {
                "group": request.group,
                "owner": request.owner,
                "stage": request.stage,
                "last_action_tick": request.tick,
                "tail_ticks": len(rewards) - request.tick,
                "undiscounted_unscaled_tail_return": sum(stream[request.tick :]),
                "omitted": not terminal,
            }
        )
    return {
        "terminal_passed_by_runtime": terminal,
        "records": len(records),
        "subjects_without_next_request": rows,
        "warning": "Per-subject returns overlap; do not sum as system reward loss.",
    }


class AuditedDQN(BaseDQN):
    def __init__(self, *args, window_terminal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_terminal = window_terminal
        # This package's V2 observations contain detached scalar/tuple data.
        # Reuse the existing lossless backend uniformly across diagnostic arms.
        self.packed_replay = True
        self.records = PackedRows()
        self.replay = {key: PackedRows() for key in self.models}

    def update(self, rewards, *, terminal, final_critic=None):
        runtime_terminal = terminal
        end_reason = getattr(rewards, "end_reason", None)
        if self.window_terminal:
            if end_reason is None:
                raise ValueError(
                    "window_terminal requires runtime end_reason provenance"
                )
            terminal = terminal or end_reason == "drain_deadline"
        audit = tail_accounting(
            self.records,
            rewards,
            terminal=terminal,
            split_stages=self.options.learning_policy.split_stages,
        )
        before = self.truncated_tails
        result = super().update(rewards, terminal=terminal, final_critic=final_critic)
        audit["actual_omitted_transitions"] = self.truncated_tails - before
        audit["terminal_passed_by_runtime"] = runtime_terminal
        audit["terminal_used_by_learner"] = terminal
        audit["end_reason"] = end_reason
        assert audit["actual_omitted_transitions"] == sum(
            row["omitted"] for row in audit["subjects_without_next_request"]
        )
        self.last_losses["v2_tail_audit"] = audit
        return result


def create_learner(
    *,
    options,
    seed,
    training=True,
    selection="sample",
    numerical_threads=1,
    window_terminal=False,
    **params,
):
    return AuditedDQN(
        options,
        seed,
        training=training,
        selection=selection,
        numerical_threads=numerical_threads,
        window_terminal=window_terminal,
    )


create_learner.CONTRACT = "smartsom.learner/v1"
create_learner.FEATURE_SCHEMA = "smartsom.candidate-features/v1"
create_learner.PARAMETER_SHARING = ("shared", "independent")
create_learner.ROLE_PARAMS = True


def create_window_learner(**kwargs):
    """Separate component identity; native optimizer parameters stay native."""
    return create_learner(window_terminal=True, **kwargs)


create_window_learner.CONTRACT = create_learner.CONTRACT
create_window_learner.FEATURE_SCHEMA = create_learner.FEATURE_SCHEMA
create_window_learner.PARAMETER_SHARING = create_learner.PARAMETER_SHARING
create_window_learner.ROLE_PARAMS = True
