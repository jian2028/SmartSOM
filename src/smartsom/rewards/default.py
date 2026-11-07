"""The preserved default overdue/failed-output objective."""


def committed_reward(outcomes, *, orders, quality_penalty):
    return -(outcomes["overdue_time"] + quality_penalty * outcomes["failed"]) / orders


class Reward:
    public_snapshots = False
    requires_events = False
    requires_actions = False

    def __init__(self, *, orders, quality_penalty):
        self.orders, self.quality_penalty = orders, quality_penalty

    def __call__(self, transition):
        return {
            "team": committed_reward(
                transition.outcomes,
                orders=self.orders,
                quality_penalty=self.quality_penalty,
            )
        }

    def begin_episode(self):
        pass

    def get_state(self):
        return {}

    def set_state(self, state):
        if state:
            raise ValueError("stateless default reward received state")


def create_reward(*, orders, quality_penalty):
    return Reward(orders=orders, quality_penalty=quality_penalty)


create_reward.CONTRACT = "smartsom.reward/v1"
