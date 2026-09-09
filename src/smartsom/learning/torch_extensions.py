"""Optional Torch feed-forward encoders and independent actor/critic branches."""

import copy
from math import prod

import torch
from torch import nn

from smartsom.config.extensions import NetworkSpec
from smartsom.learning.extensions import ObservationSpace, registration


def _activation(name):
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


def _mlp(input_size, widths, activation):
    layers = []
    current = input_size
    for width in widths:
        if type(width) is not int or width < 1:
            raise ValueError("network widths must be positive integers")
        layers.extend((nn.Linear(current, width), _activation(activation)))
        current = width
    return nn.Sequential(*layers), current


class FlattenEncoder(nn.Module):
    """Concatenate declared tensors in the public fixed space's explicit order."""

    def __init__(self, parameters, space: ObservationSpace):
        super().__init__()
        if parameters:
            raise ValueError("builtin.flatten has no parameters")
        self.space = space
        self.output_size = space.flat_size

    def forward(self, observations):
        if self.space.vector:
            expected = self.space.vector
            if tuple(observations.shape[-len(expected) :]) != expected:
                raise ValueError("encoder tensor shape disagrees with its fixed space")
            return observations.reshape(
                *observations.shape[: -len(expected)], prod(expected)
            ).float()
        if not isinstance(observations, dict) or set(observations) != {
            key for key, _ in self.space.fields
        }:
            raise ValueError("encoder Dict fields disagree with its fixed space")
        outputs, prefixes = [], set()
        for name, shape in self.space.fields:
            value = observations[name]
            if tuple(value.shape[-len(shape) :]) != shape:
                raise ValueError(
                    f"encoder tensor {name!r} shape disagrees with its fixed space"
                )
            prefix = tuple(value.shape[: -len(shape)])
            prefixes.add(prefix)
            outputs.append(value.reshape(*prefix, prod(shape)).float())
        if len(prefixes) != 1:
            raise ValueError("encoder Dict fields have inconsistent batch dimensions")
        return torch.cat(outputs, dim=-1)


class MLPEncoder(nn.Module):
    """A registered nonlinear feed-forward example usable by all three backends."""

    def __init__(self, parameters, space: ObservationSpace):
        super().__init__()
        if set(parameters) - {"hidden_sizes", "activation"}:
            raise ValueError("builtin.mlp accepts hidden_sizes and activation")
        widths = parameters.get("hidden_sizes", [32])
        activation = parameters.get("activation", "tanh")
        if (
            not isinstance(widths, (list, tuple))
            or not widths
            or activation not in ("tanh", "relu", "gelu")
        ):
            raise ValueError("invalid feed-forward encoder parameters")
        self.flatten = FlattenEncoder({}, space)
        self.network, self.output_size = _mlp(space.flat_size, widths, activation)

    def forward(self, observations):
        return self.network(self.flatten(observations))


class _Branch(nn.Module):
    def __init__(self, space, output_size, branch, provider):
        super().__init__()
        entry = registration("torch_encoder", branch.encoder, provider)
        if entry.stateful:
            raise ValueError(
                "stateful/recurrent Torch encoders are not supported; use registered feed-forward modules"
            )
        self.encoder = entry.factory(copy.deepcopy(branch.encoder.parameters), space)
        if (
            not isinstance(self.encoder, nn.Module)
            or type(getattr(self.encoder, "output_size", None)) is not int
            or self.encoder.output_size < 1
        ):
            raise ValueError(
                "Torch encoder must be nn.Module with positive output_size"
            )
        self.hidden, width = _mlp(
            self.encoder.output_size, branch.hidden_sizes, branch.activation
        )
        self.output = nn.Linear(width, output_size)

    def forward(self, observations):
        encoded = self.encoder(observations)
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.shape[-1] != self.encoder.output_size
        ):
            raise ValueError("custom encoder returned an incompatible feature tensor")
        return self.output(self.hidden(encoded))


class ActorCriticNetwork(nn.Module):
    """Actor/critic and resource roles may have different registered architectures."""

    def __init__(
        self,
        space: ObservationSpace,
        action_count: int,
        spec: NetworkSpec,
        provider: str,
        *,
        role=None,
    ):
        super().__init__()
        if type(action_count) is not int or action_count < 1:
            raise ValueError("network action count must be positive")
        selected = spec.for_role(role)
        self.actor = _Branch(space, action_count, selected.actor, provider)
        self.critic = _Branch(space, 1, selected.critic, provider)
        self.action_count = action_count

    def forward(self, observations):
        return {
            "logits": self.actor(observations),
            "values": self.critic(observations).squeeze(-1),
        }

    def masked_logits(self, observations, action_mask):
        logits = self.actor(observations)
        mask = action_mask.to(dtype=torch.bool)
        if logits.shape != mask.shape or not mask.any(dim=-1).all():
            raise ValueError("network received invalid or empty legal-action masks")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def build_actor_critic(space, action_count, spec, provider, *, role=None):
    return ActorCriticNetwork(space, action_count, spec, provider, role=role)
