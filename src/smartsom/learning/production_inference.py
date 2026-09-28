"""Load independent strategy groups and retain their independent sampling state."""

import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from smartsom.algorithms.production_rules import PolicyChoice, RulePolicy
from smartsom.config.production import named_seed
from smartsom.learning.production_models import (
    CandidateNetwork,
    PublicEncoder,
    default_network,
    tensor_inputs,
)


def read_package(source):
    source = Path(source)
    if source.is_file():
        import io
        import zipfile

        with zipfile.ZipFile(source) as archive:
            if set(archive.namelist()) - {"model.json", "weights.pt", "encoder.json"}:
                raise ValueError("unexpected component archive member")
            metadata = json.loads(archive.read("model.json"))
            weights = archive.read(metadata.get("weights_file", "weights.pt"))
            encoder_raw = archive.read("encoder.json")
            if hashlib.sha256(encoder_raw).hexdigest() != metadata["encoder_sha256"]:
                raise ValueError("model encoder hash mismatch")
            state = json.loads(encoder_raw)
        if hashlib.sha256(weights).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("component weights hash mismatch")
        weights = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    else:
        metadata = json.loads((source / "model.json").read_text())
        raw = (source / metadata["weights_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("component weights hash mismatch")
        weights = torch.load(
            source / metadata["weights_file"], map_location="cpu", weights_only=True
        )
        encoder_raw = (source / "encoder.json").read_bytes()
        if hashlib.sha256(encoder_raw).hexdigest() != metadata["encoder_sha256"]:
            raise ValueError("model encoder hash mismatch")
        state = json.loads(encoder_raw)
    return metadata, weights, state


class ModelPolicy:
    def __init__(self, encoder, network, metadata, seed, *, training=False):
        self.encoder, self.network, self.metadata = encoder, network, metadata
        self.training, self.deterministic = training, True
        self.encoder.set_training(training)
        self.network.train(training)
        self.device = next(network.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(
            seed % (2**63 - 1)
        )
        self.random = random.Random(seed)
        self.epsilon = 0.0

    def choose(self, request):
        encoded = self.encoder.encode(request)
        with torch.no_grad():
            scores, value = self.network(tensor_inputs([encoded], self.device))
            distribution = torch.distributions.Categorical(logits=scores[0])
            if (self.training or not self.deterministic) and self.metadata[
                "algorithm"
            ] == "ppo":
                index = int(
                    torch.multinomial(distribution.probs, 1, generator=self.generator)
                )
            elif self.training and self.random.random() < self.epsilon:
                index = self.random.choice(
                    [i for i, c in enumerate(request.candidates) if c.legal]
                )
            else:
                index = int(scores[0].argmax())
            logp = float(distribution.log_prob(torch.tensor(index, device=self.device)))
        return PolicyChoice(
            request.candidates[index].action,
            logp,
            float(value[0]),
            {k: v.tolist() for k, v in encoded.items()},
        )

    def value_input(self, role, owner, view):
        encoded = self.encoder.empty(role, owner, view)
        with torch.no_grad():
            _, value = self.network(tensor_inputs([encoded], self.device))
        return {k: v.tolist() for k, v in encoded.items()}, float(value[0])

    def state_dict(self):
        return {
            "generator": self.generator.get_state(),
            "random": self.random.getstate(),
            "epsilon": self.epsilon,
            "encoder": self.encoder.state_dict(),
        }

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.random.setstate(state["random"])
        self.epsilon = state["epsilon"]
        self.encoder.load_state_dict(state["encoder"])

    def fingerprint(self):
        import io

        stream = io.BytesIO()
        torch.save(self.network.state_dict(), stream)
        return hashlib.sha256(
            stream.getvalue()
            + json.dumps(self.encoder.state_dict(), sort_keys=True).encode()
        ).hexdigest()


def build_groups(prepared, *, training=True):
    config = prepared.config
    declarations = json.loads(prepared.policies_json)
    policies, learners = {}, {}
    for group, declaration in declarations.items():
        impl, role = declaration["implementation"], declaration["role"]
        seed = named_seed(config.seed, "policy:" + group)
        selected = bool(
            training and config.training and group in config.training.groups
        )
        if impl["kind"] == "rule":
            policies[group] = RulePolicy(
                role,
                impl["name"],
                seed=seed,
                parameters=impl["parameters"],
                version=impl.get("version"),
                code_sha256=impl.get("code_sha256"),
                frozen_identity=declaration.get("resolved_rule"),
            )
            continue
        if impl["kind"] == "model":
            metadata, weights, state = read_package(
                declaration["resolved_model"]["source"]
            )
            if metadata != declaration["resolved_model"]["metadata"]:
                raise ValueError("model metadata changed after configuration freeze")
        else:
            algorithm, backend = config.training.algorithm, config.training.backend
            provider = (
                "sb3.maskable_ppo"
                if backend == "sb3"
                else (
                    "rllib.ppo" if role == "central" else "rllib.resource_" + algorithm
                )
            )
            metadata = {
                "role": role,
                "algorithm": algorithm,
                "backend": backend,
                "provider": provider,
                "projection": impl["projection"],
                "observation": impl["extensions"]["observation"],
                "network": impl["extensions"]["network"] or default_network(algorithm),
            }
            weights, state = None, None
        encoder = PublicEncoder(
            prepared.scenario.factory,
            metadata["projection"],
            observation=metadata.get("observation"),
            provider=metadata["provider"],
            role=role,
            training=selected,
        )
        if state is not None:
            encoder.load_state_dict(state)
        metadata["central_private_end"] = (
            encoder.central_private_end
            if not metadata.get("observation")
            else encoder.context_size
        )
        if (
            metadata.get("candidate_width", encoder.candidate_width)
            != encoder.candidate_width
        ):
            raise ValueError("component candidate encoding dimension is incompatible")
        if metadata.get("context_size", encoder.context_size) != encoder.context_size:
            raise ValueError("component observation encoding dimension is incompatible")
        metadata["candidate_width"] = encoder.candidate_width
        parameters = json.loads(prepared.parameters_json)
        with torch.random.fork_rng():
            torch.manual_seed(seed % (2**31))
            if selected and config.training.backend == "rllib":
                from smartsom.learning.production_rllib_v3 import build_learner

                learner = build_learner(
                    encoder.context_size,
                    metadata["network"],
                    metadata["algorithm"],
                    parameters,
                    central=role == "central",
                    central_private_end=metadata["central_private_end"],
                    device=config.runtime.device,
                    candidate_width=encoder.candidate_width,
                )
                network = learner.module["default_policy"].network
                learners[group] = learner
            elif selected:
                from smartsom.learning.production_sb3_v3 import build_sb3

                learner = build_sb3(
                    encoder.context_size,
                    metadata["network"],
                    parameters,
                    seed % (2**31),
                    metadata["central_private_end"],
                    config.runtime.device,
                    candidate_width=encoder.candidate_width,
                )
                learners[group], network = learner, learner.policy.network
            else:
                network = CandidateNetwork(
                    encoder.context_size,
                    metadata["network"],
                    metadata["provider"],
                    metadata["algorithm"],
                    central=role == "central",
                    central_private_end=metadata["central_private_end"],
                    candidate_width=encoder.candidate_width,
                )
        network.to(config.runtime.device)
        if weights is not None:
            network.load_state_dict(weights, strict=True)
        if selected and metadata["algorithm"] == "dqn":
            learners[group].module["default_policy"].target.load_state_dict(
                network.state_dict()
            )
        if not selected:
            network.requires_grad_(False)
        policies[group] = ModelPolicy(
            encoder, network, copy.deepcopy(metadata), seed, training=selected
        )
    return policies, learners
