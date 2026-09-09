import pytest

from smartsom.config.codec import canonical_json
from smartsom.config.experiment import SearchOptions
from smartsom.experiments.search import OptunaSession, require_optuna


def options(**overrides):
    return SearchOptions.model_validate_json(
        canonical_json(
            {
                "method": "optuna",
                "seed": 42,
                "trials": 12,
                "space": {
                    "algorithm.learning_rate": {
                        "type": "float",
                        "low": 0.0001,
                        "high": 0.01,
                        "log": True,
                    }
                },
                "objective": "makespan",
                "failure_policy": "all_complete",
                **overrides,
            }
        )
    )


@pytest.fixture
def optuna():
    # This optional integration module fails explicitly when the search extra is
    # requested; base-only test environments report a deliberate skip.
    import os

    if os.environ.get("SMARTSOM_REQUIRE_SEARCH") == "1":
        return require_optuna()
    return pytest.importorskip("optuna")


def test_real_seeded_tpe_is_repeatable_across_session_recreation(tmp_path, optuna):
    observations = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        values = []
        for index in range(12):
            session = OptunaSession(root, options())
            parameters, trial = session.ask(index)
            value = parameters["algorithm.learning_rate"]
            values.append(value)
            session.finish(trial, "completed", (value - 0.004) ** 2)
        observations.append(values)
    assert observations[0] == observations[1]
    assert len(set(observations[0])) > 1
    study = session._study(12)
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert len(study.trials) == 12
    assert all(trial.state.name == "COMPLETE" for trial in study.trials)


def test_pending_proposal_is_recovered_and_failed_retry_keeps_parameters(
    tmp_path, optuna
):
    session = OptunaSession(tmp_path, options())
    parameters, first = session.ask(0)
    restored = OptunaSession(tmp_path, options())
    assert restored.ask(0) == (parameters, first)
    restored.finish(first, "interrupted", None)
    assert restored.ask(0) == (parameters, first)
    restored.finish(first, "failed", None)
    retry_parameters, retry = restored.ask(0, parameters=parameters)
    assert retry_parameters == parameters
    assert retry["number"] != first["number"]
    restored.finish(retry, "completed", 3.0)
    assert [trial.state.name for trial in restored._study(1).trials] == [
        "FAIL",
        "COMPLETE",
    ]


def test_pruning_is_explicit_and_uses_reported_validation_steps(tmp_path, optuna):
    session = OptunaSession(tmp_path, options(pruning=True))
    for index in range(5):
        _, trial = session.ask(index)
        assert not session.report(trial, 1.0, 1)
        session.finish(trial, "completed", 1.0)
    _, candidate = session.ask(5)
    assert session.report(candidate, 10.0, 1)
    session.finish(candidate, "pruned", None)
    assert session._study(6).trials[-1].state.name == "PRUNED"
    off_root = tmp_path / "off"
    off_root.mkdir()
    disabled = OptunaSession(off_root, options(pruning=False))
    _, trial = disabled.ask(0)
    assert not disabled.report(trial, 999999.0, 1)
    assert disabled._study(1).trials[0].intermediate_values == {}


def test_structured_categorical_choices_use_categorical_distributions(tmp_path, optuna):
    session = OptunaSession(
        tmp_path,
        options(
            space={
                "algorithm.hidden_sizes": {
                    "type": "categorical",
                    "choices": [[32, 32], [64, 64]],
                },
            }
        ),
    )
    value, identity = session.ask(0)
    assert list(value["algorithm.hidden_sizes"]) in [[32, 32], [64, 64]]
    _, trial = session.trial(identity)
    assert isinstance(
        trial.distributions["algorithm.hidden_sizes"],
        optuna.distributions.CategoricalDistribution,
    )
