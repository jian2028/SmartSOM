"""Human-authored scenes expand once into the existing physical contracts.

This module does not run policies or import Qt. Explicit resource identities and
positions remain visible; defaults apply only to non-geometric parameters.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from smartsom.config.codec import StrictModel, primitive, read_document
from smartsom.config.factory_design import (
    FactoryAuthoring,
    parse_factory_design_file,
)

SCHEMA = "smartsom.scenario/v1"
MACHINE_PARAMETERS = {"processing_rate_multiplier", "quality_modes", "breakdown"}
AGV_PARAMETERS = {"job_capacity", "move_cells_per_tick", "initial_heading", "battery"}
BUFFER_ROLES = {"machine_pre", "machine_post"}


class SceneModel(StrictModel):
    model_config = ConfigDict(
        strict=False, extra="forbid", frozen=True, validate_default=True
    )


def merge(defaults, overrides):
    """Mappings merge by field; lists and scalars replace explicitly."""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(result.get(key), dict)
            and isinstance(value, dict)
            and not ("mode" in value and value["mode"] != result[key].get("mode"))
            else deepcopy(value)
        )
    return result


def keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context}: unknown fields {sorted(unknown)}")


class Orders(SceneModel):
    total: int = Field(default=128, gt=0)
    initial: int = Field(default=16, ge=0)


class Arrivals(SceneModel):
    mean_interval_ticks: float = Field(default=36, gt=0)


class Routes(SceneModel):
    families: list[list[str]] = Field(
        default_factory=lambda: [list(s) for s in ("ABCD", "BCDA", "CDAB", "DABC")]
    )
    repeat_first_probability: float = Field(default=0.5, ge=0, le=1)


class Processing(SceneModel):
    base_ticks: dict[str, float] = Field(
        default_factory=lambda: dict(A=30, B=40, C=50, D=40)
    )
    relative_variation: float = Field(default=0.2, gt=0, lt=1)


class DueDates(SceneModel):
    urgent_probability: float = Field(default=0.2, ge=0, le=1)
    normal_multiplier: tuple[float, float] = (2, 3)
    urgent_multiplier: tuple[float, float] = (1.5, 2.5)


class JobGenerator(SceneModel):
    orders: Orders = Field(default_factory=Orders)
    arrivals: Arrivals = Field(default_factory=Arrivals)
    routes: Routes = Field(default_factory=Routes)
    processing: Processing = Field(default_factory=Processing)
    due_dates: DueDates = Field(default_factory=DueDates)

    def workload_fields(self):
        return dict(
            schema="smartsom.base-workload/v1",
            orders=self.orders.total,
            initial_orders=self.orders.initial,
            mean_interarrival=self.arrivals.mean_interval_ticks,
            families=self.routes.families,
            optional_repeat_probability=self.routes.repeat_first_probability,
            base_ticks=self.processing.base_ticks,
            workload_rho=self.processing.relative_variation,
            urgent_probability=self.due_dates.urgent_probability,
            normal_due=self.due_dates.normal_multiplier,
            urgent_due=self.due_dates.urgent_multiplier,
        )

    @classmethod
    def from_workload(cls, workload):
        return cls(
            orders=Orders(total=workload.orders, initial=workload.initial_orders),
            arrivals=Arrivals(mean_interval_ticks=workload.mean_interarrival),
            routes=Routes(
                families=workload.families,
                repeat_first_probability=workload.optional_repeat_probability,
            ),
            processing=Processing(
                base_ticks=workload.base_ticks, relative_variation=workload.workload_rho
            ),
            due_dates=DueDates(
                urgent_probability=workload.urgent_probability,
                normal_multiplier=workload.normal_due,
                urgent_multiplier=workload.urgent_due,
            ),
        )


class ScenarioFile(SceneModel):
    schema_id: Literal["smartsom.scenario/v1"] = Field(alias="schema")
    factory_spec: dict = Field(alias="factory")
    jobs: JobGenerator = Field(default_factory=JobGenerator)
    authoring: FactoryAuthoring = Field(default_factory=FactoryAuthoring)

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, value):
        from smartsom.config.base_diagnostic import MachineFaults

        if not isinstance(value, dict) or not isinstance(value.get("factory"), dict):
            return value
        value = deepcopy(value)
        factory = value["factory"]
        builtins = {
            "machines": {
                "processing_rate_multiplier": "1",
                "quality_modes": [
                    {"quality_mode_id": "normal", "time_scale": "1", "error_rate": "0"}
                ],
                "breakdown": MachineFaults().model_dump(mode="json"),
            },
            "agvs": {
                "job_capacity": 1,
                "move_cells_per_tick": 1,
                "initial_heading": "east",
                "battery": None,
            },
        }
        for section, defaults in builtins.items():
            if isinstance(factory.get(section), dict):
                supplied = factory[section].get("defaults", {})
                if isinstance(supplied, dict):
                    factory[section]["defaults"] = merge(defaults, supplied)
        if isinstance(factory.get("buffer_defaults", {}), dict):
            factory["buffer_defaults"] = merge(
                {
                    role: {"storage": {"mode": "pool", "capacity": 4}}
                    for role in sorted(BUFFER_ROLES)
                },
                factory.get("buffer_defaults", {}),
            )
        return value

    @model_validator(mode="after")
    def valid(self):
        try:
            self.expand()
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid or missing scene field: {exc}") from exc
        return self

    @property
    def factory(self):
        return self.expand()[0].factory

    @property
    def reliability(self):
        # Legacy Studio envelopes expose this property; scene faults are in the
        # machine defaults/overrides and are never silently imported from it.
        return None

    def expand(self):
        from smartsom.config.base_diagnostic import BaseWorkload, MachineFaults

        data = deepcopy(self.factory_spec)
        machines = data.pop("machines")
        agvs = data.pop("agvs")
        buffer_defaults = data.pop("buffer_defaults", {})
        keys(buffer_defaults, BUFFER_ROLES, "buffer_defaults")
        for role, spec in buffer_defaults.items():
            keys(spec, {"storage"}, f"buffer_defaults.{role}")
        for name, section, allowed in (
            ("machines", machines, MACHINE_PARAMETERS),
            ("agvs", agvs, AGV_PARAMETERS),
        ):
            keys(section, {"defaults", "items"}, name)
            keys(section.get("defaults", {}), allowed, f"{name}.defaults")
            if not isinstance(section.get("items"), list):
                raise ValueError(f"{name}.items must be an explicit list")
        defaults = machines.get("defaults", {})
        fault_defaults = MachineFaults.model_validate(defaults.get("breakdown", {}))
        faults = {}
        data["machines"] = []
        for item in machines["items"]:
            keys(item, {"id", "name", "operation", "position", "overrides"}, "machine")
            overrides = item.get("overrides", {})
            keys(overrides, MACHINE_PARAMETERS, f"machine {item.get('id')}")
            effective = merge(defaults, overrides)
            rule = MachineFaults.model_validate(effective.pop("breakdown", {}))
            if rule != fault_defaults or not rule.enabled:
                faults[item["id"]] = rule.model_dump(mode="json")
            if not isinstance(item["operation"], str) or not item["operation"]:
                raise ValueError("each machine requires one explicit operation")
            keys(item["position"], {"x", "y"}, "machine.position")
            data["machines"].append(
                dict(
                    machine_id=item["id"],
                    name=item.get("name", item["id"]),
                    operation_types=[item["operation"]],
                    footprint={**item["position"], "width": 1, "height": 1},
                    **effective,
                )
            )
        data["agvs"] = []
        for item in agvs["items"]:
            keys(item, {"id", "name", "position", "overrides"}, "AGV")
            keys(item.get("overrides", {}), AGV_PARAMETERS, f"AGV {item.get('id')}")
            keys(item["position"], {"x", "y"}, "AGV.position")
            data["agvs"].append(
                dict(
                    agv_id=item["id"],
                    name=item.get("name", item["id"]),
                    initial_cell=item["position"],
                    **merge(agvs.get("defaults", {}), item.get("overrides", {})),
                )
            )
        data["buffers"] = [
            merge(buffer_defaults.get(b.get("role"), {}), b)
            for b in data.get("buffers", [])
        ]
        envelope = parse_factory_design_file(
            dict(
                schema="smartsom.factory/v2",
                factory=data,
                authoring=primitive(self.authoring),
            )
        )
        workload = BaseWorkload.model_validate(
            {
                **self.jobs.workload_fields(),
                "fault_intervals": fault_defaults.intervals,
                "repair": fault_defaults.repair,
                "machine_faults": faults,
            }
        )
        return envelope, workload

    @classmethod
    def from_factory(cls, envelope, workload=None):
        from smartsom.config.base_diagnostic import BaseWorkload

        if envelope.reliability is not None:
            raise ValueError(
                "Legacy factory reliability needs explicit migration; it is not the current workload fault protocol"
            )
        workload = workload or BaseWorkload.model_validate(
            JobGenerator().workload_fields()
        )
        data = primitive(asdict(envelope.factory))
        for machine in data["machines"]:
            f = machine["footprint"]
            if (f["width"], f["height"]) != (1, 1):
                raise ValueError(
                    f"{machine['machine_id']}: scenes require 1x1 machines; old geometry was not resized"
                )
            if len(machine["operation_types"]) != 1:
                raise ValueError(
                    f"{machine['machine_id']}: scenes require exactly one operation"
                )
        machine_defaults = {
            "processing_rate_multiplier": "1",
            "quality_modes": [
                {"quality_mode_id": "normal", "time_scale": "1", "error_rate": "0"}
            ],
            **common(data["machines"], MACHINE_PARAMETERS - {"breakdown"}),
        }
        machine_defaults["breakdown"] = dict(
            enabled=True,
            intervals={
                k: v.model_dump(mode="json")
                for k, v in workload.fault_intervals.items()
            },
            repair=workload.repair.model_dump(mode="json"),
        )
        agv_defaults = {
            "job_capacity": 1,
            "move_cells_per_tick": 1,
            "initial_heading": "east",
            "battery": None,
            **common(data["agvs"], AGV_PARAMETERS),
        }
        machines = []
        for m in data["machines"]:
            item = dict(
                id=m["machine_id"],
                name=m["name"],
                operation=m["operation_types"][0],
                position={k: m["footprint"][k] for k in ("x", "y")},
            )
            overrides = difference(
                machine_defaults,
                {k: v for k, v in m.items() if k in MACHINE_PARAMETERS},
            )
            if m["machine_id"] in workload.machine_faults:
                overrides["breakdown"] = workload.machine_faults[
                    m["machine_id"]
                ].model_dump(mode="json")
            if overrides:
                item["overrides"] = overrides
            machines.append(item)
        agvs = []
        for a in data["agvs"]:
            item = dict(id=a["agv_id"], name=a["name"], position=a["initial_cell"])
            overrides = difference(
                agv_defaults, {k: v for k, v in a.items() if k in AGV_PARAMETERS}
            )
            if overrides:
                item["overrides"] = overrides
            agvs.append(item)
        data["machines"] = dict(defaults=machine_defaults, items=machines)
        data["agvs"] = dict(defaults=agv_defaults, items=agvs)
        data["buffer_defaults"] = {}
        for role in BUFFER_ROLES:
            group = [b for b in data["buffers"] if b["role"] == role]
            defaults = common(group, {"storage"}) or {
                "storage": {"mode": "pool", "capacity": 4}
            }
            data["buffer_defaults"][role] = defaults
            for b in group:
                if b.get("storage") == defaults.get("storage"):
                    b.pop("storage", None)
        return cls(
            schema=SCHEMA,
            factory=data,
            jobs=JobGenerator.from_workload(workload),
            authoring=envelope.authoring,
        )

    def with_design(self, design, authoring=None, *, renames=None):
        """Reconcile geometric/property edits without flattening defaults."""
        data = deepcopy(self.factory_spec)
        renames = renames or {}
        expanded = primitive(asdict(design))
        previous_design = primitive(asdict(self.factory))
        from pydantic import TypeAdapter

        from smartsom.config.codec import canonical_json
        from smartsom.domain.factory_design import AGVDesign, MachineDesign

        for section, id_field, allowed in (
            ("machines", "machine_id", MACHINE_PARAMETERS - {"breakdown"}),
            ("agvs", "agv_id", AGV_PARAMETERS),
        ):
            old = {
                renames.get(item["id"], item["id"]): item
                for item in data[section]["items"]
            }
            defaults = data[section].get("defaults", {})
            resources = expanded.pop(section)
            prior_values = {
                renames.get(r[id_field], r[id_field]): r
                for r in previous_design[section]
            }
            default_values = {}
            if resources:
                # Compare typed values, so YAML 1.0 and Decimal "1" do not create
                # accidental overrides just by opening/saving the editor.
                raw_default = {
                    **resources[0],
                    **{k: v for k, v in defaults.items() if k in allowed},
                }
                model = MachineDesign if section == "machines" else AGVDesign
                default_values = primitive(
                    asdict(
                        TypeAdapter(model).validate_json(
                            canonical_json(raw_default), strict=True
                        )
                    )
                )
            items = []
            for resource in resources:
                identifier = resource[id_field]
                previous = old.get(identifier, {})
                item = dict(id=identifier, name=resource["name"])
                if section == "machines":
                    f = resource["footprint"]
                    if (f["width"], f["height"]) != (1, 1) or len(
                        resource["operation_types"]
                    ) != 1:
                        raise ValueError(
                            "Machines require 1x1 cells and one explicit operation"
                        )
                    item.update(
                        operation=resource["operation_types"][0],
                        position={k: f[k] for k in ("x", "y")},
                    )
                else:
                    item["position"] = resource["initial_cell"]
                overrides = deepcopy(previous.get("overrides", {}))
                for key in allowed:
                    value = resource.get(key)
                    if value == prior_values.get(identifier, default_values).get(key):
                        continue
                    if value != default_values.get(key) or key in overrides:
                        overrides[key] = value
                if overrides:
                    item["overrides"] = overrides
                items.append(item)
            data[section]["items"] = items
        old_buffers = {
            renames.get(b["buffer_id"], b["buffer_id"]): b
            for b in data.get("buffers", [])
        }
        prior_buffers = {
            renames.get(b["buffer_id"], b["buffer_id"]): b
            for b in previous_design["buffers"]
        }
        for b in expanded.get("buffers", []):
            prior = prior_buffers.get(b["buffer_id"])
            default = data.get("buffer_defaults", {}).get(b["role"], {}).get("storage")
            if (
                prior is not None
                and prior["role"] == b["role"]
                and prior["storage"]["mode"] == b["storage"]["mode"]
            ):
                # Preserve authored omissions and explicit pins; only actual
                # property edits become new overrides. A mode change replaces
                # the whole storage definition instead of merging its fields.
                storage = deepcopy(old_buffers[b["buffer_id"]].get("storage", {}))
                storage.update(
                    (key, value)
                    for key, value in b["storage"].items()
                    if value != prior["storage"].get(key)
                )
                if storage:
                    b["storage"] = storage
                else:
                    b.pop("storage")
            elif prior is None and b["storage"] == default:
                b.pop("storage")
        data.update(expanded)
        return type(self)(
            schema=SCHEMA,
            factory=data,
            jobs=self.jobs,
            authoring=authoring or self.authoring,
        )

    def inherit_new(self, design, identifiers):
        """Placement uses the document defaults; paste preserves its own values."""
        from smartsom.config.factory_design import parse_factory_design_file

        data = primitive(asdict(design))
        for section, id_field in (("machines", "machine_id"), ("agvs", "agv_id")):
            defaults = {
                k: v
                for k, v in self.factory_spec[section]["defaults"].items()
                if k != "breakdown"
            }
            for resource in data[section]:
                if resource[id_field] in identifiers:
                    resource.update(deepcopy(defaults))
        for resource in data["buffers"]:
            if resource["buffer_id"] in identifiers:
                resource.update(
                    deepcopy(
                        self.factory_spec.get("buffer_defaults", {}).get(
                            resource["role"], {}
                        )
                    )
                )
        return parse_factory_design_file(
            dict(schema="smartsom.factory/v2", factory=data)
        ).factory


def common(items, fields):
    import json

    result = {}
    for key in sorted(fields):
        values = [json.dumps(i[key], sort_keys=True) for i in items if key in i]
        if values:
            result[key] = json.loads(Counter(values).most_common(1)[0][0])
    return result


def difference(defaults, values):
    return {k: deepcopy(v) for k, v in values.items() if defaults.get(k) != v}


def load_scene(path):
    import hashlib

    raw = Path(path).read_bytes()
    import yaml

    from smartsom.config.codec import _UniqueLoader

    return ScenarioFile.model_validate(
        yaml.load(raw, Loader=_UniqueLoader)
    ), hashlib.sha256(raw).hexdigest()


def expand_documents(documents):
    """Authoring is one scene; execution keeps one existing physical input path."""
    scene = ScenarioFile.model_validate(documents.pop("scenario"))
    factory, workload = scene.expand()
    documents["factory"] = primitive(factory)
    documents["workload"] = workload.model_dump(mode="json", by_alias=True)
    ex = documents["experiment"]
    ex.pop("scenario")
    ex.update(factory="factory.yaml", workload="workload.yaml")
    return scene


def load_studio_file(path):
    from smartsom.config.factory_design import load_factory_design_file

    if read_document(path).get("schema") == SCHEMA:
        return load_scene(path)
    envelope, digest = load_factory_design_file(path)
    return ScenarioFile.from_factory(envelope), digest


def save_scene(path, scene, *, expected_digest=None):
    import yaml

    from smartsom.config.factory_design import save_authoring_bytes

    class SceneDumper(yaml.SafeDumper):
        pass

    def mapping(dumper, value):
        compact = len(value) <= 4 and all(
            not isinstance(v, (dict, list, tuple)) for v in value.values()
        )
        return dumper.represent_mapping(
            "tag:yaml.org,2002:map", value, flow_style=compact
        )

    def sequence(dumper, value):
        compact = len(value) <= 12 and all(
            not isinstance(v, (dict, list, tuple)) for v in value
        )
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", value, flow_style=compact
        )

    SceneDumper.add_representer(dict, mapping)
    SceneDumper.add_representer(list, sequence)
    scene = ScenarioFile.model_validate(primitive(scene))
    payload = (
        "# SmartSOM scenario. Time: ticks; positions: grid cells.\n"
        "# Machines and AGVs occupy exactly one cell.\n"
        + yaml.dump(
            primitive(scene), Dumper=SceneDumper, sort_keys=False, allow_unicode=True
        )
    ).encode()
    return save_authoring_bytes(path, payload, expected_digest=expected_digest)
