"""Frozen factory fault schedules and the model's H calculation."""

from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic import ValidationError

from smartsom.config.codec import ConfigurationError, primitive
from smartsom.config.factory_design import (
    FactoryDesignFile,
    load_factory_design_file,
    save_factory_design,
    save_factory_design_file,
)
from smartsom.config.reliability import (
    FactoryReliability,
    factory_heterogeneity,
    factory_reliability_outages,
    validate_factory_conditions,
)
from smartsom.domain.factory_design import (
    FactoryDesign,
    Footprint,
    GridDesign,
    MachineDesign,
)
from smartsom.domain.quality import QualityMode
from smartsom.studio.document import FactoryDocument
from smartsom.studio.persistence import RecoveryStore


def document(reliability=None, *, rates=(1, 1), errors=(".03", ".03")):
    machines = tuple(
        MachineDesign(
            f"machine_{i}",
            f"Machine {i}",
            Footprint(i, 2, 1, 1),
            operation_types=("drill",),
            processing_rate_multiplier=Decimal(str(rate)),
            quality_modes=tuple(
                QualityMode(name, Decimal(time), Decimal(error) + offset)
                for name, time, offset in (
                    ("slow", "1.2", Decimal("-.02")),
                    ("normal", "1", Decimal(0)),
                    ("fast", ".8", Decimal(".02")),
                )
            ),
        )
        for i, (rate, error) in enumerate(zip(rates, errors, strict=True), 1)
    )
    return FactoryDesignFile(
        schema="smartsom.factory/v2",
        factory=FactoryDesign(
            "factory",
            "Factory",
            GridDesign(8, 5),
            operation_types=("drill",),
            machines=machines,
        ),
        reliability=None
        if reliability is None
        else FactoryReliability.model_validate(reliability),
    )


def uniform():
    return {
        "enabled": True,
        "defaults": {
            "uptime": {"distribution": "uniform", "min_ticks": 4, "max_ticks": 4},
            "repair": {"min_ticks": 3, "max_ticks": 3},
        },
        "machines": {"machine_2": {"enabled": False}},
    }


def test_absent_reliability_retains_old_primitive_and_yaml(tmp_path):
    original = document()
    assert "reliability" not in primitive(original)
    path = tmp_path / "factory.yaml"
    save_factory_design_file(path, original)
    assert "reliability:" not in path.read_text()
    loaded, _ = load_factory_design_file(path)
    assert loaded == original


def test_uniform_schedule_counts_idle_uptime_and_clips_horizon():
    original = document(uniform())
    outages = factory_reliability_outages(original, 99, 17)
    assert [(x.machine_id, x.start, x.end) for x in outages] == [
        ("machine_1", 4, 7),
        ("machine_1", 11, 14),
    ]
    assert factory_reliability_outages(original, 99, 6)[0].end == 6
    assert (
        primitive(original)["reliability"]
        == primitive(document(uniform()))["reliability"]
    )


def test_machine_override_replaces_only_selected_distribution():
    value = uniform()
    value["machines"]["machine_2"] = {
        "uptime": {"distribution": "uniform", "min_ticks": 1, "max_ticks": 1}
    }
    outages = factory_reliability_outages(document(value), 99, 9)
    assert [(x.start, x.end) for x in outages if x.machine_id == "machine_2"] == [
        (1, 4),
        (5, 8),
    ]
    assert [(x.start, x.end) for x in outages if x.machine_id == "machine_1"] == [
        (4, 7)
    ]


def test_exponential_is_deterministic_and_policy_factory_name_independent():
    value = uniform()
    value["defaults"]["uptime"] = {"distribution": "exponential", "mean_ticks": 5.0}
    original = document(value)
    first = factory_reliability_outages(original, 202, 60)
    assert first and first == factory_reliability_outages(original, 202, 60)
    assert first != factory_reliability_outages(original, 203, 60)
    changed = original.model_copy(
        update={
            "factory": replace(original.factory, factory_id="new_name", name="New name")
        }
    )
    assert first == factory_reliability_outages(changed, 202, 60)
    assert all(left.end < right.start for left, right in zip(first, first[1:]))
    assert all(row.end <= 60 and row.start >= 1 for row in first)


def test_extreme_finite_exponential_mean_is_outside_finite_horizon():
    value = uniform()
    value["defaults"]["uptime"] = {"distribution": "exponential", "mean_ticks": 1e308}
    assert factory_reliability_outages(document(value), 202, 60) == ()


def test_machine_stream_does_not_depend_on_document_order_or_other_rules():
    value = uniform()
    value["defaults"]["uptime"] = {"distribution": "exponential", "mean_ticks": 5.0}
    first = document(value)
    value["machines"]["machine_2"] = {"enabled": True}
    second = document(value)
    second = second.model_copy(
        update={
            "factory": replace(
                second.factory, machines=tuple(reversed(second.factory.machines))
            )
        }
    )
    assert factory_reliability_outages(first, 13, 100) == tuple(
        row
        for row in factory_reliability_outages(second, 13, 100)
        if row.machine_id == "machine_1"
    )


def test_master_switch_suppresses_overrides():
    value = uniform()
    value["enabled"] = False
    value["machines"]["machine_2"] = {"enabled": True}
    assert factory_reliability_outages(document(value), 1, 20) == ()


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "bad_id",
        "bool_ticks",
        "negative",
        "inverted",
        "empty",
        "missing",
        "nan",
        "repair_zero",
    ],
)
def test_strict_reliability_validation(change):
    value = uniform()
    if change == "unknown":
        value["defaults"]["hidden"] = True
    elif change == "bad_id":
        value["machines"] = {"does_not_exist": {"enabled": False}}
    elif change == "bool_ticks":
        value["defaults"]["uptime"]["min_ticks"] = True
    elif change == "negative":
        value["defaults"]["uptime"]["min_ticks"] = -1
    elif change == "inverted":
        value["defaults"]["uptime"]["min_ticks"] = 5
    elif change == "empty":
        value["machines"] = {"machine_1": {}}
    elif change == "missing":
        value["defaults"] = None
    elif change == "nan":
        value["defaults"]["uptime"] = {
            "distribution": "exponential",
            "mean_ticks": float("nan"),
        }
    else:
        value["defaults"]["repair"]["min_ticks"] = 0
    with pytest.raises((ValidationError, ConfigurationError)):
        document(value)


@pytest.mark.parametrize(
    "seed,horizon", [(-1, 10), (True, 10), (1, 0), (1, True), (2**64, 10)]
)
def test_schedule_arguments_are_strict(seed, horizon):
    with pytest.raises(ConfigurationError):
        factory_reliability_outages(document(uniform()), seed, horizon)


def test_full_document_studio_save_and_recovery_preserve_reliability(tmp_path):
    original = document(uniform())
    path = tmp_path / "factory.yaml"
    digest = save_factory_design_file(path, original)
    doc = FactoryDocument(
        original.factory, path, digest, reliability=original.reliability
    )
    doc.design = replace(doc.design, name="Edited in Studio")
    assert doc.file.reliability == original.reliability
    recovery = RecoveryStore(tmp_path / "studio")
    recovered, _ = load_factory_design_file(recovery.snapshot(doc))
    assert recovered.reliability == original.reliability
    # Existing callers saving just a design must also retain complete file data.
    save_factory_design(path, doc.design, expected_digest=digest)
    saved, _ = load_factory_design_file(path)
    assert saved.reliability == original.reliability
    assert saved.factory.name == "Edited in Studio"


def test_md_homogeneous_and_g1_examples():
    report = factory_heterogeneity(document())
    assert report.supported and report.h_speed == report.h_quality == report.h == 0
    report = factory_heterogeneity(
        document(rates=(1.1, 0.9), errors=(".03225", ".02725"))
    )
    assert report.h_speed == pytest.approx(0.1)
    assert report.h_quality == pytest.approx(0.2487468593)
    assert report.h == pytest.approx(0.1895718861)
    assert all(
        row["weighted_quality"] == pytest.approx(row["reference_quality"])
        for row in report.components
    )


def test_md_g4_normal_component_and_faults_do_not_enter_h():
    first = document(rates=(1.3, 0.7), errors=(".0349", ".0209"))
    report = factory_heterogeneity(first)
    normal = next(row for row in report.components if row["mode"] == "normal")
    assert normal["speed_squared"] == pytest.approx(0.09)
    assert normal["quality_squared"] == pytest.approx(0.4459)
    assert report.h == pytest.approx(0.5176388693)
    with_faults = first.model_copy(
        update={"reliability": document(uniform()).reliability}
    )
    assert factory_heterogeneity(with_faults) == report


def test_md_conditions_freeze_geometry_capabilities_and_nominal_yield():
    baseline = document()
    changed = document(rates=(1.1, 0.9), errors=(".03225", ".02725"))
    reports = validate_factory_conditions((baseline, changed))
    assert reports[0].h == 0 and reports[1].h > 0
    bad_rate = document(rates=(1.1, 1))
    with pytest.raises(ConfigurationError, match="nominal rate"):
        validate_factory_conditions((baseline, bad_rate))
    bad_quality = document(errors=(".04", ".04"))
    with pytest.raises(ConfigurationError, match="capacity-weighted quality"):
        validate_factory_conditions((baseline, bad_quality))
    moved = changed.model_copy(
        update={
            "factory": replace(
                changed.factory,
                machines=(
                    replace(
                        changed.factory.machines[0], footprint=Footprint(1, 3, 1, 1)
                    ),
                    changed.factory.machines[1],
                ),
            )
        }
    )
    with pytest.raises(ConfigurationError, match="geometry"):
        validate_factory_conditions((baseline, moved))


def test_legacy_modes_receive_diagnostic_without_fabricated_h():
    value = document()
    value = value.model_copy(
        update={
            "factory": replace(
                value.factory,
                machines=tuple(
                    replace(
                        machine,
                        quality_modes=(QualityMode("normal", Decimal(1), Decimal(0)),),
                    )
                    for machine in value.factory.machines
                ),
            )
        }
    )
    report = factory_heterogeneity(value)
    assert not report.supported and report.h is None and report.diagnostic
    assert validate_factory_conditions((value,)) == (report,)
