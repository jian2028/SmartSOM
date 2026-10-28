"""Stateful custom centralized policy using semantic candidate identities."""


class Policy:
    def __init__(self):
        self.decisions = 0

    def choose(self, request):
        self.decisions += 1
        # The final candidate always advances physical time.
        return request.candidates[0].identity

    def get_state(self):
        return {"decisions": self.decisions}

    def set_state(self, state):
        self.decisions = state["decisions"]


def create_policy(*, seed):
    return Policy()
