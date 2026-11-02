"""DQN output construction over shared candidate features."""

from smartsom.learning.heads import CandidateHead


def create_head(width, *, empty_wait=False):
    return CandidateHead(width, empty_wait=empty_wait, value_head=False)


def create_legacy_rule_head(width, *, empty_wait=False):
    """Preserve the archived rule-DQN initialization and dormant-head gradients.

    All five roles historically constructed critic then wait, even with WAIT
    disabled. Their RNG draws affect subsequent roles and the later social heads.
    The frozen critic never participates in TD loss or Adam; masked wait remains
    in the forward graph, preserving zero gradients instead of missing gradients.
    """
    head = CandidateHead(width, empty_wait=True, value_head=True)
    head.critic.requires_grad_(False)
    return head
