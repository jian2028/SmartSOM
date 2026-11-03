"""C1-inspired central rule selection on the current reservation protocol."""

from math import isfinite


def validate_parameters(
    *,
    learning_rate=0.0003,
    epochs=3,
    batch_size=256,
    clip=0.2,
    entropy=0.01,
    learning_reward_scale=0.001,
):
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1
        for x in (epochs, batch_size)
    ):
        raise ValueError("epochs and batch_size must be positive integers")
    if not all(
        isfinite(x) for x in (learning_rate, clip, entropy, learning_reward_scale)
    ):
        raise ValueError("central PPO parameters must be finite")
    if (
        learning_rate <= 0
        or not 0 < clip < 1
        or entropy < 0
        or learning_reward_scale <= 0
    ):
        raise ValueError("invalid central PPO parameters")


def create_learner(
    *, options, seed, training=True, selection="sample", numerical_threads=1, **params
):
    from .learner import CentralPPO

    validate_parameters(**params)
    return CentralPPO(
        options,
        seed,
        training=training,
        selection=selection,
        numerical_threads=numerical_threads,
        **params,
    )


def ownership(options):
    return {
        "central": {
            "mode": "global",
            "entities": sorted({x for xs in options._owner_ids.values() for x in xs}),
            "networks": 1,
            "actor_networks": 1,
            "critic_networks": 1,
            "parameters": dict(options.learning_policy.learner.params),
        }
    }


create_learner.CONTRACT = "smartsom.learner/v1"
create_learner.PARAMETER_SHARING = ("shared",)
create_learner.validate_parameters = validate_parameters
create_learner.network_ownership = ownership
