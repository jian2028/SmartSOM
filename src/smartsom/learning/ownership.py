"""Stable network ownership; role templates are copied without new RNG draws."""

from copy import deepcopy

from torch import nn

from smartsom.learning.subjects import groups, model_groups


class NetworkOwnership:
    def __init__(self, options):
        self.options = {g: options.for_role(g) for g in groups(options)}
        sharing = options.learning_policy.parameter_sharing
        self.keys = {}
        self.owners = {}
        self.templates = {}
        self.split_stages = options.learning_policy.split_stages
        self.frozen_templates = {
            point.replace(".", "__") for point in options.learning_policy.frozen_stages
        }
        for group, template, factory_group in model_groups(options):
            independent = sharing.mode(group) == "independent"
            owners = tuple(sorted(options._owner_ids.get(group, ())))
            if independent and not owners:
                raise ValueError(
                    f"{group}: independent networks require a compiled factory owner inventory"
                )
            keys = (
                tuple(f"{template}__{owner.encode().hex()}" for owner in owners)
                if independent
                else (template,)
            )
            self.keys[group] = (*self.keys.get(group, ()), *keys)
            self.templates.update({key: template for key in keys})
            if independent:
                self.owners.update(
                    {
                        (factory_group, owner): key
                        for owner, key in zip(owners, keys, strict=True)
                    }
                )
        self.contract = {
            g: {
                (role + "/" + owner if g in self.split_stages else owner): key
                for (role, owner), key in self.owners.items()
                if role == g or role.startswith(g + ".")
            }
            or {"shared": self.keys[g] if g in self.split_stages else g}
            for g in groups(options)
        }

    def is_frozen(self, key):
        return self.templates[key] in self.frozen_templates

    def key(self, request):
        group = request.group
        template = f"{group}__{request.stage}" if group in self.split_stages else group
        factory_group = (
            f"{group}.{request.stage}" if group in self.split_stages else group
        )
        if template in self.keys[group]:
            return template
        try:
            return self.owners[factory_group, request.owner]
        except KeyError as exc:
            raise ValueError(
                f"unknown network owner: {request.group}/{request.owner}"
            ) from exc

    def expand(self, templates):
        return nn.ModuleDict(
            {
                key: templates[self.templates[key]]
                if key == self.templates[key]
                else deepcopy(templates[self.templates[key]])
                for group in self.keys
                for key in self.keys[group]
            }
        ).eval()

    def validate(self, contract):
        if contract != self.contract:
            raise ValueError("checkpoint/sampler network ownership mismatch")
