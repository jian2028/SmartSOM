"""Example custom learning entry sharing runtime and complete PPO state contracts.

Replace update() to research a learner variant; retain physical-time records,
legal candidates and the state contract. No simulator or tracking code is copied.
"""

from smartsom.learning.ppo.learner import BasePPO


class ResearchPPO(BasePPO):
    def update(self, rewards, *, terminal, final_critic):
        result = super().update(rewards, terminal=terminal, final_critic=final_critic)
        self.last_losses = {**result, "custom": {"batch_ticks": len(rewards)}}
        return self.last_losses


def create_learner(
    *, options, seed, training=True, selection="sample", numerical_threads=1, **params
):
    return ResearchPPO(
        options,
        seed,
        training=training,
        selection=selection,
        numerical_threads=numerical_threads,
    )


create_learner.CONTRACT = "smartsom.learner/v1"
