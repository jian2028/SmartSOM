"""Stable network ownership; role templates are copied without new RNG draws."""

from copy import deepcopy

from torch import nn

from smartsom.domain.base_decisions import GROUPS


class NetworkOwnership:
    def __init__(self, options):
        self.options = {g: options.for_role(g) for g in GROUPS}
        sharing = options.learning_policy.parameter_sharing
        self.keys = {}
        self.owners = {}
        for group in GROUPS:
            independent = sharing.mode(group) == "independent"
            owners = tuple(sorted(options._owner_ids.get(group, ())))
            if independent and not owners:
                raise ValueError(
                    f"{group}: independent networks require a compiled factory owner inventory"
                )
            self.keys[group] = (
                tuple(f"{group}__{owner.encode().hex()}" for owner in owners)
                if independent
                else (group,)
            )
            if independent:
                self.owners.update(
                    {
                        (group, owner): key
                        for owner, key in zip(owners, self.keys[group], strict=True)
                    }
                )
        self.contract = {
            g: {owner: key for (role, owner), key in self.owners.items() if role == g}
            or {"shared": g}
            for g in GROUPS
        }

    def key(self, request):
        if self.keys[request.group] == (request.group,):
            return request.group
        try:
            return self.owners[request.group, request.owner]
        except KeyError as exc:
            raise ValueError(
                f"unknown network owner: {request.group}/{request.owner}"
            ) from exc

    def expand(self, templates):
        return nn.ModuleDict(
            {
                key: templates[group] if key == group else deepcopy(templates[group])
                for group in GROUPS
                for key in self.keys[group]
            }
        ).eval()

    def validate(self, contract):
        if contract != self.contract:
            raise ValueError("checkpoint/sampler network ownership mismatch")
