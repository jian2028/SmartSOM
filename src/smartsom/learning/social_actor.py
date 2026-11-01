"""Actor-only, candidate-linked attention and same-input pooling controls."""

import math

import torch
from torch import nn

from smartsom.domain.base_decisions import FEATURES
from smartsom.domain.social import PEER_KINDS


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, outputs))


class Pool(nn.Module):
    def __init__(self, width, query_size, mode):
        super().__init__()
        self.mode, self.width = mode, width
        if mode == "attention":
            self.query = nn.Linear(query_size, width)
            self.key = nn.Linear(width, width)
            self.value = nn.Linear(width, width)
        else:
            self.combine = nn.Linear(2 * width, width)

    def forward(self, values, mask, query):
        # Arbitrary leading batch dimensions; never softmax an all-masked row.
        shape = values.shape[:-2]
        values = values.reshape(-1, values.shape[-2], self.width)
        mask = mask.reshape(-1, mask.shape[-1])
        query = query.reshape(-1, query.shape[-1])
        nonempty = mask.any(-1)
        result = values.new_zeros((len(values), self.width))
        if nonempty.any():
            v, m = values[nonempty], mask[nonempty]
            if self.mode == "attention":
                q = self.query(query[nonempty])
                scores = (self.key(v) * q[:, None]).sum(-1) / math.sqrt(self.width)
                weights = scores.masked_fill(~m, -torch.inf).softmax(-1)
                result[nonempty] = (self.value(v) * weights[..., None]).sum(-2)
            else:
                mean = (v * m[..., None]).sum(-2) / m.sum(-1, keepdim=True)
                maximum = v.masked_fill(~m[..., None], -torch.inf).max(-2).values
                result[nonempty] = self.combine(torch.cat((mean, maximum), -1))
        return result.reshape(*shape, self.width)


class SocialActor(nn.Module):
    def __init__(self, width, group, stages, mode):
        super().__init__()
        self.group, self.stages = group, stages
        self.peer = mlp(FEATURES + len(PEER_KINDS), width, width)
        self.peer_pool = Pool(width, 2 * width + FEATURES, mode)
        if group != "dispatcher":
            self.link = mlp(FEATURES + width + 5, width, width)
            self.link_pool = Pool(width, 2 * width, mode)
        self.heads = nn.ModuleDict({str(s): mlp(3 * width, width, 1) for s in stages})
        if group in ("pre", "post"):
            self.stop = mlp(2 * width, width, 1)
        if group == "dispatcher" and 0 in stages:
            self.source = mlp(2 * width, width, width)
            self.source_pool = Pool(width, width, mode)
            self.wait = mlp(2 * width, width, 1)

    def forward(self, batch, candidate, context, base_logits):
        active = torch.zeros_like(batch["stage"], dtype=torch.bool)
        for stage in self.stages:
            active |= batch["stage"] == stage
        if not torch.equal(active, batch["social_active"]):
            raise ValueError("social observation does not match enabled role/stage")
        if not active.any():
            return base_logits
        social = batch["social"]
        own = context[:, None].expand(-1, candidate.shape[1], -1)
        query = torch.cat((candidate, own), -1)
        meta = social["meta"]
        kinds = meta[:, None, :3]
        kinds = kinds.expand(*social["peers"].shape[:-1], 3)
        encoded = self.peer(torch.cat((social["peers"], kinds), -1))
        rows, columns, slots = social["indices"].unbind(-1)
        link_query = torch.cat((query[rows, columns], social["query"]), -1)
        peer_context = self.peer_pool(encoded, social["peer_mask"], link_query)
        if self.group == "dispatcher":
            linked = torch.zeros_like(candidate)
            linked[rows, columns] = peer_context
        else:
            links = self.link(torch.cat((social["query"], peer_context, meta), -1))
            # Pad only the small H-dimensional summaries, not 48-field peer sets
            # across every candidate's largest (typically STOP) link inventory.
            summaries = links.new_zeros(*social["link_mask"].shape, links.shape[-1])
            summaries[rows, columns, slots] = links
            linked = self.link_pool(summaries, social["link_mask"], query)
        combined = torch.cat((query, linked), -1)
        logits = base_logits
        for stage in self.stages:
            selected = self.heads[str(stage)](combined).squeeze(-1)
            logits = torch.where((batch["stage"] == stage)[:, None], selected, logits)
        if self.group in ("pre", "post"):
            scores = self.stop(torch.cat((own, linked), -1)).squeeze(-1)
            logits = torch.where(batch["stop"] & active[:, None], scores, logits)
        if self.group == "dispatcher" and 0 in self.stages:
            sources = self.source(torch.cat((candidate, linked), -1))
            mask = batch["mask"] & ~batch["empty_wait"]
            summary = self.source_pool(sources, mask, context)
            scores = self.wait(torch.cat((context, summary), -1))
            logits = torch.where(batch["empty_wait"] & active[:, None], scores, logits)
        return logits


def pack_social(requests, candidates, count_scale):
    n = len(requests)
    entries = []
    for r in requests:
        by_id = {c.identity: c.links for c in r.social or ()}
        if r.social is not None and (
            len(by_id) != len(r.social)
            or set(by_id) != {c.identity for c in r.candidates}
        ):
            raise ValueError("social candidates must match semantic action identities")
        entries.append([by_id.get(c.identity, ()) for c in r.candidates])
    links = max(1, max(len(rows) for e in entries for rows in e))
    peers = max(
        1,
        max(
            (len(link.peers) for e in entries for rows in e for link in rows), default=0
        ),
    )
    total_links = sum(len(rows) for entry in entries for rows in entry)
    values = torch.zeros(total_links, peers, FEATURES)
    peer_mask = torch.zeros(total_links, peers, dtype=torch.bool)
    link_mask = torch.zeros(n, candidates, links, dtype=torch.bool)
    indices = torch.zeros(total_links, 3, dtype=torch.long)
    query = torch.zeros(total_links, FEATURES)
    meta = torch.zeros(total_links, 5)
    index = 0
    for i, entry in enumerate(entries):
        for j, rows in enumerate(entry):
            for k, link in enumerate(rows):
                query[index] = torch.tensor(link.query)
                indices[index] = torch.tensor((i, j, k))
                link_mask[i, j, k] = True
                meta[index, PEER_KINDS.index(link.kind)] = 1
                meta[index, 3] = link.terminal
                meta[index, 4] = link.count / count_scale
                if link.peers:
                    values[index, : len(link.peers)] = torch.tensor(link.peers)
                    peer_mask[index, : len(link.peers)] = True
                index += 1
    return {
        "peers": values,
        "peer_mask": peer_mask,
        "link_mask": link_mask,
        "query": query,
        "meta": meta,
        "indices": indices,
    }
