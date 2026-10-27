"""Minimal deterministic provider over detached semantic candidates."""


class FirstCandidate:
    def choose(self, request):
        return request.candidates[0].identity

    def get_state(self):
        return {}

    def set_state(self, state):
        if state != {}:
            raise ValueError("unexpected provider state")


def create_policy(*, seed):
    return FirstCandidate()
