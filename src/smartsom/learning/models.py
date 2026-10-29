"""Compose selected observation encoders and algorithm-specific output heads."""

from torch import nn

from smartsom.config.components import construct, instantiate, resolve
from smartsom.domain.base_decisions import GROUPS
from smartsom.representations.base.batching import pack


class EncodedModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    @property
    def critic(self):
        return self.head.critic

    def actor(self, batch):
        return self.head(batch, self.encoder(batch))

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def make_models(options, *, value_head=True):
    if options.network is not None:
        return nn.ModuleDict(
            {
                g: construct(
                    options.network, group=g, options=options, value_head=value_head
                )
                for g in GROUPS
            }
        )
    if value_head:
        from smartsom.learning.ppo.heads import create_head
    else:
        from smartsom.learning.dqn.heads import create_head
    factory = resolve(options.encoder)
    if getattr(factory, "OUTPUT_SCHEMA", None) != "smartsom.candidate-features/v1":
        raise ValueError("encoder output does not match candidate heads")
    models = nn.ModuleDict()
    for group in GROUPS:
        encoder = instantiate(
            factory, options.encoder.params, group=group, options=options
        )
        models[group] = EncodedModel(
            encoder,
            create_head(
                encoder.width,
                empty_wait=options.allow_empty_wait and group == "dispatcher",
            ),
        )
    if options.social_arm != "base":
        from smartsom.domain.social import SOCIAL_ARMS
        from smartsom.learning.social_actor import SocialActor

        for group in GROUPS:
            stages = tuple(
                sorted(
                    {
                        {
                            "process": 0,
                            "pickup": 0,
                            "delivery": 1,
                            "accept": 0,
                            "service": 1,
                        }[s]
                        for g, s in SOCIAL_ARMS[options.social_arm]
                        if g == group
                    }
                )
            )
            if stages:
                models[group].head.social = SocialActor(
                    models[group].encoder.width, group, stages, options.social_pool
                )
    return models


def packer(options):
    spec = options.network or options.encoder
    factory = resolve(spec)
    prepare = getattr(factory, "pack", None)
    if not callable(prepare):
        raise ValueError("network/encoder factory requires pack")
    if spec.target in {
        "smartsom.representations.base.encoder.create_encoder",
        "smartsom.representations.social.encoder.create_encoder",
    }:
        return pack

    def custom(requests, critics, count_scale, encoded=None, **kwargs):
        return prepare(requests, critics=critics, options=options)

    return custom
