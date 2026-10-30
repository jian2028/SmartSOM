"""Detached current-state social relations; no policy/framework dependency."""

from dataclasses import dataclass

SOCIAL_ARMS = {
    "base": (),
    "machine": (("machine", "process"),),
    "dispatcher-pickup": (("dispatcher", "pickup"),),
    "dispatcher-delivery": (("dispatcher", "delivery"),),
    "dispatcher": (("dispatcher", "pickup"), ("dispatcher", "delivery")),
    "pre": (("pre", "accept"), ("pre", "service")),
    "post": (("post", "accept"), ("post", "service")),
    "input": (("input", "accept"), ("input", "service")),
}
SOCIAL_ARMS["all"] = tuple(
    pair
    for arm in ("machine", "dispatcher", "pre", "post", "input")
    for pair in SOCIAL_ARMS[arm]
)
PEER_KINDS = ("resource", "station", "vehicle")


@dataclass(frozen=True, slots=True)
class SocialLink:
    identity: str
    kind: str
    query: tuple[float, ...]
    peers: tuple[tuple[float, ...], ...]
    terminal: bool = False
    count: int = 0


@dataclass(frozen=True, slots=True)
class SocialCandidate:
    identity: str
    links: tuple[SocialLink, ...]
