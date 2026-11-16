"""Public Ray Tune phase resource callback; no admission ledger or calibration."""

from ray.tune.execution.placement_groups import PlacementGroupFactory

BOOTSTRAP_BUNDLES = [{"CPU": 1.0}]


def phase_allocation(controller, trial, result, scheduler):
    """Called by the stock ResourceChangingScheduler after a durable boundary.

    Every trial begins with the same lightweight bootstrap request. The
    checkpoint owns the next phase; a restored trial must negotiate again if
    Tune restored the bootstrap request. No model is loaded before that check.
    """
    if result.get("done"):
        return None
    if result.get("control_boundary"):
        return PlacementGroupFactory(BOOTSTRAP_BUNDLES)
    phase = result["next_phase"]
    config = trial.config.get("trial", trial.config)
    bundles = config["phase_bundles"][phase]
    return PlacementGroupFactory(bundles)


def phase_resources_ready(current, bundles):
    """Check the actual placement group before starting any phase work."""
    return current == PlacementGroupFactory(bundles)
