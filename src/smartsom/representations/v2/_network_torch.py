"""V2 shared candidate scorers with implicit social attention or direct concatenation."""

import hashlib
from contextlib import contextmanager

import torch
from torch import nn

from smartsom.representations.v2 import concat
from smartsom.representations.v2.observation import CONTEXT, FEATURES, SCHEMA
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import social_config


@contextmanager
def initialization(name):
    seed = int.from_bytes(
        hashlib.sha256(f"{torch.initial_seed()}/{name}".encode()).digest()[:8], "big"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def mlp(name, sizes, *, final_relu=True):
    with initialization(name):
        layers = []
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if final_relu or index < len(sizes) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)


class Attention(nn.Module):
    def __init__(self, name, query, peer):
        super().__init__()
        self.encoder = mlp(name + "/peer", [peer, 32, 32])
        with initialization(name + "/qkv"):
            self.query = nn.Linear(query, 32)
            self.key = nn.Linear(32, 32)
            self.value = nn.Linear(32, 32)

    def forward(self, query, peers, mask):
        encoded = self.encoder(peers)
        q = self.query(query).unsqueeze(-2)
        scores = (q * self.key(encoded)).sum(-1) / (32**0.5)
        # Empty sets get a finite softmax and exactly zero output.
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, -1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        return (weights.unsqueeze(-1) * self.value(encoded)).sum(-2)


class Branch(nn.Module):
    def __init__(self, group, stage, social_enabled, concat_slots=None):
        super().__init__()
        self.group, self.stage, self.social_enabled = group, stage, social_enabled
        self.concat_slots = concat_slots
        point = group + "." + stage
        self.candidate = mlp(
            point + "/candidate", [FEATURES[point] + len(CATALOG[point]), 32, 32]
        )
        width = 32 if point == "dispatcher.pickup" else 67 if group == "machine" else 64
        extra = 0
        if concat_slots is not None:
            extra = concat.width(group, stage, concat_slots)
        elif social_enabled:
            if group == "dispatcher":
                self.peers = Attention(point + "/vehicles", width, 3)
                extra = 32
            elif group in ("post", "input") and stage == "accept":
                condition = 4 if group == "post" else 3
                self.job = mlp(point + "/job", [condition, 32, 32])
                self.downstream = Attention(point + "/downstream", 64, 9)
                self.job_relation = mlp(point + "/relation", [66, 32, 32])
                # Job relations are already learned 32-dimensional embeddings;
                # the second attention must not add another peer MLP.
                self.jobs = Attention(point + "/jobs", width, 32)
                self.jobs.encoder = nn.Identity()
                extra = 33 if group == "post" else 32
            else:
                self.downstream = Attention(point + "/downstream", width, 9)
                extra = 34
                if group == "pre" and stage == "accept":
                    self.alternatives = Attention(point + "/alternatives", width, 9)
                    extra += 33
        self.score = mlp(
            point
            + (
                "/concat-score"
                if concat_slots is not None
                else "/isl-score"
                if social_enabled
                else "/base-score"
            ),
            [width + extra, 64, 32, 1],
            final_relu=False,
        )

    def forward(self, batch, common):
        candidate = self.candidate(batch["candidates"])
        parts = (
            [candidate]
            if common is None
            else [common.unsqueeze(1).expand(-1, candidate.shape[1], -1), candidate]
        )
        if self.group == "machine":
            parts.append(batch["modes"])
        base = torch.cat(parts, -1)
        parts = [base]
        if self.concat_slots is not None:
            parts.append(
                concat.features(batch, self.group, self.stage, self.concat_slots)
            )
        elif self.social_enabled:
            if self.group == "dispatcher":
                parts.append(self.peers(base, batch["peers"], batch["peer_mask"]))
            elif self.group in ("post", "input") and self.stage == "accept":
                jobs = self.job(batch["jobs"])
                # RLlib carries owner supply without a candidate axis. Native
                # packing may still carry a distinct supply for each candidate.
                shared_supply = jobs.ndim == 3
                owner = (
                    common[:, None, :] if shared_supply else common[:, None, None, :]
                )
                query = torch.cat([owner.expand(*jobs.shape[:-1], 32), jobs], -1)
                downstream = self.downstream(
                    query, batch["job_peers"], batch["job_peer_mask"]
                )
                related = self.job_relation(
                    torch.cat([jobs, downstream, batch["job_meta"]], -1)
                )
                job_mask = batch["job_mask"]
                job_count = batch["job_count"]
                if shared_supply:
                    related = related.unsqueeze(1).expand(-1, base.shape[1], -1, -1)
                    job_mask = job_mask.unsqueeze(1).expand(-1, base.shape[1], -1)
                    job_count = job_count.unsqueeze(1).expand(-1, base.shape[1], -1)
                parts.append(self.jobs(base, related, job_mask))
                if self.group == "post":
                    parts.append(job_count)
            else:
                parts.extend(
                    [
                        self.downstream(
                            base, batch["downstream"], batch["downstream_mask"]
                        ),
                        batch["downstream_meta"],
                    ]
                )
                if self.group == "pre" and self.stage == "accept":
                    parts.extend(
                        [
                            self.alternatives(
                                base, batch["alternatives"], batch["alternatives_mask"]
                            ),
                            batch["alternatives_count"],
                        ]
                    )
        return (
            self.score(torch.cat(parts, -1))
            .squeeze(-1)
            .masked_fill(~batch["mask"], -torch.inf)
        )


class Network(nn.Module):
    def __init__(self, group, *, social_enabled, value_head, concat_slots=None):
        super().__init__()
        role, _, selected_stage = group.partition(".")
        self.group = role
        stages = [
            point.split(".")[1]
            for point in CATALOG
            if point.startswith(role + ".")
            and (not selected_stage or point.endswith("." + selected_stage))
        ]
        if role == "dispatcher" and not selected_stage:
            raise ValueError("V2 Dispatcher requires split_stages: [dispatcher]")
        common_width = CONTEXT.get(group, CONTEXT.get(role))
        self.common = (
            mlp(group + "/common", [common_width, 32, 32]) if common_width else None
        )
        self.branches = nn.ModuleDict(
            {
                stage: Branch(role, stage, social_enabled, concat_slots)
                for stage in stages
            }
        )
        if value_head:
            with initialization(group + "/critic"):
                self.critic = nn.Sequential(
                    nn.Linear(48, 64), nn.Tanh(), nn.Linear(64, 1)
                )

    def actor(self, batch):
        reference = next(iter(batch["stages"].values()))["context"]
        scores = reference.new_full((batch["size"], batch["width"]), -torch.inf)
        for stage, data in batch["stages"].items():
            common = self.common(data["context"]) if self.common is not None else None
            values = self.branches[stage](data, common)
            scores = scores.index_copy(0, data["indices"], values)
        return scores

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def _peers(relations, width, *, field="peers"):
    from smartsom.representations.v2._packing import peers_array

    return peers_array(relations, width, arrays=torch, field=field)


def pack(requests, *, critics, options):
    from smartsom.representations.v2._packing import pack as pack_arrays

    return pack_arrays(requests, critics=critics, options=options, arrays=torch)


def create_network(*, group, options, value_head, social_enabled=None, social=None):
    settings = social_config(social, social_enabled)
    # The paired built-ins must describe the same information and fusion layout.
    # External observations retain their own schema contract.
    if (
        options.observation.target
        == "smartsom.representations.v2.observation.create_observation"
    ):
        observed = social_config(**options.observation.params)
        if observed != settings:
            raise ValueError("observation/network social configuration mismatch")
    return Network(
        group,
        social_enabled=settings.kind == "implicit",
        value_head=value_head,
        concat_slots=settings.slots,
    )


create_network.INPUT_SCHEMA = SCHEMA
create_network.pack = pack
