"""Explicit DQN factory; Torch remains optional until invoked."""


def create_learner(
    *, options, seed, training=True, selection="sample", numerical_threads=1, **params
):
    from .learner import BaseDQN

    return BaseDQN(
        options,
        seed,
        training=training,
        selection=selection,
        numerical_threads=numerical_threads,
    )


create_learner.CONTRACT = "smartsom.learner/v1"
create_learner.FEATURE_SCHEMA = "smartsom.candidate-features/v1"

create_learner.PARAMETER_SHARING = ("shared", "independent")
create_learner.ROLE_PARAMS = True
