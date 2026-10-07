"""Passive, bounded V3 service observations, never a missed-action penalty."""


class PickupDiagnostics:
    def __init__(self, event_context=None, *, history_complete=True):
        from smartsom.experiments.progress_diagnostics import initial_state

        progress = initial_state(event_context)
        progress["history_complete"] = history_complete
        self.values = {
            "progress_state": progress,
            "schema": "smartsom.pickup-diagnostics/v2",
            "semantics": "nonexclusive-intentions/3",
            "matching": "first_arrival",
            "history_complete": history_complete,
            "scope": "one environment episode; aggregate over physical owners",
            "phase": "decision request / admitted Buffer service / committed pickup event",
            "interpretation": "descriptive opportunities and completed-service latency; not errors",
            "eligible_pickup_boundaries": None,
            "eligible_empty_agv_decisions": None,
            "missed_pickup_boundaries": None,
            "first_reservation_tick": None,
            "legacy_status": "unavailable: NO_REQUEST and exclusive reservations absent",
            "empty_target_decisions": 0,
            "ready_source_choice_opportunities": 0,
            "ready_source_choices": 0,
            "nonready_source_choices_when_ready_available": 0,
            "admitted_service_slots": 0,
            "service_opportunity_boundaries": 0,
            "pickup_started": 0,
            "first_pickup_tick": None,
            "arrival_to_pickup_ticks_sum": 0,
            "arrival_to_pickup_observations": 0,
            "arrival_to_pickup_ticks_max": None,
        }

    def observe(self, coordinator, outcome):
        from smartsom.experiments.progress_diagnostics import observe

        if hasattr(getattr(coordinator, "sim", None), "machine_state"):
            observe(self.values["progress_state"], coordinator, outcome)
        values = self.values
        slots = {}
        for decision in coordinator.records:
            if decision["role"] == "buffer":
                # One Buffer prefix produces several conditional records, but
                # count is the source-local admitted capacity, not one per row.
                slots[decision["owner"]] = decision["count"]
            if decision["role"] != "dispatcher":
                continue
            view = decision["observation"]
            vehicle = view["agvs"][decision["owner"]]
            if vehicle["job"] is not None:
                continue
            values["empty_target_decisions"] += 1
            ready = {
                row["identity"]
                for row in decision["candidates"]
                if row["legal"]
                and isinstance(row["action"], dict)
                and view.get("sources", {}).get(row["action"]["owner"], {}).get("ready")
            }
            if ready:
                values["ready_source_choice_opportunities"] += 1
                selected = decision["candidate"] in ready
                values["ready_source_choices"] += int(selected)
                values["nonready_source_choices_when_ready_available"] += int(
                    not selected
                )
        values["admitted_service_slots"] += sum(slots.values())
        values["service_opportunity_boundaries"] += int(bool(slots))
        for event in outcome.get("events", ()):
            if event["kind"] != "pickup_started":
                continue
            values["pickup_started"] += 1
            if values["first_pickup_tick"] is None:
                values["first_pickup_tick"] = event["tick"]
            arrived = coordinator.sim.agvs[event["agv"]].get("arrived_at")
            if arrived is not None and arrived <= event["tick"]:
                delay = event["tick"] - arrived
                values["arrival_to_pickup_ticks_sum"] += delay
                values["arrival_to_pickup_observations"] += 1
                values["arrival_to_pickup_ticks_max"] = max(
                    values["arrival_to_pickup_ticks_max"] or 0, delay
                )

    def summary(self):
        import copy

        result = copy.deepcopy(self.values)
        count = result["arrival_to_pickup_observations"]
        result["arrival_to_pickup_ticks_mean"] = (
            result["arrival_to_pickup_ticks_sum"] / count if count else None
        )
        opportunities = result["ready_source_choice_opportunities"]
        result["ready_source_choice_fraction"] = (
            result["ready_source_choices"] / opportunities if opportunities else None
        )
        return result
