"""PPO output construction over shared candidate features."""

from smartsom.learning.heads import CandidateHead


def create_head(width, *, empty_wait=False):
    return CandidateHead(width, empty_wait=empty_wait, value_head=True)
