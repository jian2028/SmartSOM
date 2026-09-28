"""One boundary coordinator shared by rules, inference and learning collectors."""

import copy

from smartsom.algorithms.pickup_matching import MATCHING_RULES, matching_rng
from smartsom.algorithms.production_rules import PolicyChoice
from smartsom.config.codec import primitive


class BoundaryCoordinator:
    def __init__(
        self, sim, policies, bindings, matching="global_optimal", *, episode=0
    ):
        if sim.protocol is None:
            raise ValueError("composition requires the v3 physical protocol")
        if matching not in MATCHING_RULES:
            raise ValueError("unknown pickup matching rule")
        self.sim, self.policies, self.bindings = sim, policies, bindings
        self.matching, self.episode = matching, episode
        sim.protocol.episode = episode
        self.records = []

    def group_for(self, role, owner):
        binding = self.bindings[role]
        return binding.get("overrides", {}).get(owner, binding["default"])

    def select(self, request):
        group = self.group_for(request.role, request.owner)
        policy = self.policies[group]
        choice = policy.choose(request)
        if not isinstance(choice, PolicyChoice):
            choice = PolicyChoice(choice)
        legal = [c for c in request.candidates if c.legal]
        if choice.action not in [c.action for c in legal]:
            raise ValueError("policy returned an illegal semantic proposal")
        candidate = next(c for c in legal if c.action == choice.action)
        record = {
            "tick": request.tick,
            "stage": request.stage,
            "role": request.role,
            "owner": request.owner,
            "group": group,
            "prefix": list(request.prefix),
            "count": request.count,
            "candidates": primitive(request.candidates),
            "observation": copy.deepcopy(request.observation),
            "proposal": primitive(choice.action),
            "candidate": candidate.identity,
            "log_probability": choice.log_probability,
            "value": choice.value,
            "actor_mask": not request.deterministic,
            "model_input": primitive(choice.model_input),
        }
        self.records.append(record)
        return choice.action

    def tick(self, *, stage_only=False):
        protocol = self.sim.protocol
        self.records = []
        try:
            requests = protocol.begin()
            machines, dispatchers = {}, {}
            for request in requests:
                action = self.select(request)
                (machines if request.role == "machine" else dispatchers)[
                    request.owner
                ] = action
            buffer_requests = protocol.accept_proposals(machines, dispatchers)
            prefixes = {}
            for request in buffer_requests:
                prefix = []
                while len(prefix) < request.count:
                    current = protocol.buffer_request(request.owner, prefix)
                    prefix.append(self.select(current))
                prefixes[request.owner] = tuple(prefix)
            pairs = []
            matching_records = []
            for owner, prefix in prefixes.items():
                vehicles = protocol.service_vehicles[owner]
                rng = matching_rng(
                    self.sim.scenario.seed,
                    self.episode,
                    owner,
                    self.sim.tick,
                    prefix,
                    vehicles,
                )
                matcher = MATCHING_RULES[self.matching]
                kwargs = (
                    {"inspection_tie": protocol.inspection_tie}
                    if self.matching == "priority_greedy"
                    else {}
                )
                selected = matcher(
                    prefix, vehicles, protocol.matching_cost, rng, **kwargs
                )
                pairs.extend(selected)
                matching_records.append(
                    {
                        "source": owner,
                        "rule": self.matching,
                        "prefix": list(prefix),
                        "vehicles": sorted(vehicles),
                        "pairs": primitive(selected),
                    }
                )
            movers = {}
            for request in protocol.prepare_services(prefixes, pairs):
                movers[request.owner] = self.select(request)
            protocol.log = list(self.records)
            if stage_only:
                protocol.abort()
                return {"decisions": list(self.records)}
            result = protocol.commit(movers)
            result["matching"] = matching_records
            return result
        except BaseException:
            protocol.abort()
            raise

    def state_dict(self):
        return {group: policy.state_dict() for group, policy in self.policies.items()}

    def load_state_dict(self, state):
        for group, value in state.items():
            self.policies[group].load_state_dict(value)


def command_from_record(data):
    from smartsom.domain.production_decisions import BoundaryCommand, DispatchTarget

    return BoundaryCommand(
        tuple((k, tuple(v)) for k, v in data["machines"]),
        tuple(
            (k, DispatchTarget(**v) if v is not None else None)
            for k, v in data["dispatchers"]
        ),
        tuple((k, tuple(v)) for k, v in data["prefixes"]),
        tuple(tuple(v) for v in data["matching"]),
        tuple(tuple(v) for v in data["movers"]),
        data["contract"],
    )


def replay_boundary(sim, record):
    command = command_from_record(record["actions"])
    saved = list(record.get("decisions", []))
    if saved:
        protocol = sim.protocol
        index = 0

        def verify(request):
            nonlocal index
            if index >= len(saved):
                raise ValueError("missing recorded decision")
            actual = saved[index]
            expected = {
                "tick": request.tick,
                "stage": request.stage,
                "role": request.role,
                "owner": request.owner,
                "prefix": list(request.prefix),
                "count": request.count,
                "candidates": primitive(request.candidates),
                "observation": request.observation,
            }
            if any(
                primitive(actual.get(k)) != primitive(v) for k, v in expected.items()
            ):
                raise ValueError(
                    "recorded candidate/observation differs from physical phase"
                )
            candidates = [
                c
                for c in request.candidates
                if c.identity == actual["candidate"] and c.legal
            ]
            if (
                len(candidates) != 1
                or primitive(candidates[0].action) != actual["proposal"]
            ):
                raise ValueError(
                    "recorded semantic proposal differs from its legal candidate"
                )
            index += 1

        try:
            for request in protocol.begin():
                verify(request)
            requests = protocol.accept_proposals(
                dict(command.machines), dict(command.dispatchers)
            )
            prefixes = dict(command.prefixes)
            for request in requests:
                prefix = []
                for job in prefixes[request.owner]:
                    verify(protocol.buffer_request(request.owner, prefix))
                    prefix.append(job)
            for request in protocol.prepare_services(prefixes, command.matching):
                verify(request)
            if index != len(saved):
                raise ValueError("unexpected recorded decision")
            protocol.log = saved
            result = protocol.commit(dict(command.movers))
        except BaseException:
            protocol.abort()
            raise
    else:
        result = sim.step(command)
    for key in ("state", "events", "reward", "rejections"):
        if primitive(result[key]) != primitive(record[key]):
            raise ValueError(f"v3 execution audit mismatch: {key}")
    return result
