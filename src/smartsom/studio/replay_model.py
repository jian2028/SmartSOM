"""Read-only presentation adapters for current recordings.

No simulator or model is executed here. Missing evidence stays missing.
"""

from dataclasses import dataclass

from smartsom.studio.replay_evidence import movement_conflicts, resource_conflicts

LANES = (
    ("movement", "Movement conflict", "#d94848"),
    ("resource", "Pickup/drop conflict", "#bf7b21"),
    ("delivery", "Qualified delivery", "#24836c"),
    ("scrap", "Scrap", "#788492"),
    ("inspection", "Inspection events", "#8b67bb"),
    ("outage", "Machine outage", "#b58a2c"),
)


@dataclass(frozen=True)
class ReplayEvent:
    tick: int
    kind: str
    owners: frozenset[str]
    detail: str
    lane: str | None = None
    end: int | None = None


def event_owners(event, state, previous):
    owners = set()
    for key in (
        "agv",
        "agent_id",
        "machine",
        "machine_id",
        "owner",
        "station",
        "station_id",
        "source",
        "demand",
        "job",
        "job_id",
        "actor",
    ):
        value = event.get(key)
        if isinstance(value, str):
            owners.add(value)
            if key == "actor" and ":" in value:
                owners.add(value.split(":", 1)[1])
    for job in (*(event.get("jobs") or ()), event.get("job"), event.get("job_id")):
        if not isinstance(job, str):
            continue
        owners.add(job)
        for frame in (previous, state):
            data = frame.get("jobs", {}).get(job, {})
            owners.update(
                str(data[k]) for k in ("demand", "location", "holder") if data.get(k)
            )
            for agv, vehicle in frame.get("agvs", {}).items():
                if vehicle.get("job") == job:
                    owners.add(agv)
    return frozenset(owners)


class ReplayIndex:
    """Event index and bounded per-selection history from a read-only recording."""

    def __init__(self, recording, factory):
        self.recording, self.factory = recording, factory
        self.ports = {p.port_id: p for p in factory.ports}
        self.events = []
        self.last_tick = recording.last_tick
        previous, open_outages = {}, {}
        for tick in range(recording.last_tick + 1):
            row = getattr(recording, "summary_row", recording.row)(tick)
            state = row["state"]
            for lane, owners in (
                ("movement", movement_conflicts(row)),
                ("resource", resource_conflicts(row)),
            ):
                for owner in sorted(owners):
                    self.events.append(
                        ReplayEvent(
                            tick,
                            "conflict",
                            frozenset((owner,)),
                            f"{owner} · {lane} conflict · action rejected",
                            lane,
                        )
                    )
            for actor, reason in row.get("rejections", {}).items():
                if reason != "conflict":
                    self.events.append(
                        ReplayEvent(
                            tick,
                            "action_rejected",
                            event_owners({"actor": actor}, state, previous),
                            f"{actor} · rejected: {reason}",
                        )
                    )
            for event in row.get("events", ()):
                kind = event.get("kind", "event")
                lane = None
                if kind in ("trash", "automatic_disposal", "scrapped"):
                    lane = "scrap"
                elif kind in (
                    "inspection_started",
                    "inspection_completed",
                    "quality_revealed",
                ):
                    lane = "inspection"
                elif kind == "output" and event.get("quality") == "PASS":
                    lane = "delivery"
                owners = event_owners(event, state, previous)
                # Avoid leaking hidden quality or dumping arbitrary trace payloads.
                detail = " · ".join(
                    (kind, ", ".join(sorted(owners)), str(event.get("reason", "")))
                ).strip(" ·")
                self.events.append(ReplayEvent(tick, kind, owners, detail, lane))
            for demand in sorted(
                set(state.get("completed", ())) - set(previous.get("completed", ()))
            ):
                owners = {demand}
                for jid, job in state.get("jobs", {}).items():
                    if job.get("demand") == demand:
                        owners.update(event_owners({"job": jid}, state, previous))
                self.events.append(
                    ReplayEvent(
                        tick,
                        "qualified_delivery",
                        frozenset(owners),
                        f"Qualified delivery · {demand}",
                        "delivery",
                    )
                )
            for key, machine in state.get("machines", {}).items():
                down = bool(machine.get("down") or machine.get("status") == "DOWN")
                was = bool(
                    previous.get("machines", {}).get(key, {}).get("down")
                    or previous.get("machines", {}).get(key, {}).get("status") == "DOWN"
                )
                if down and not was:
                    open_outages[key] = tick
                if was and not down and key in open_outages:
                    start = open_outages.pop(key)
                    self.events.append(
                        ReplayEvent(
                            start,
                            "outage",
                            frozenset((key,)),
                            f"{key} · outage {start}–{tick}",
                            "outage",
                            tick,
                        )
                    )
            previous = state
        for key, start in open_outages.items():
            self.events.append(
                ReplayEvent(
                    start,
                    "outage",
                    frozenset((key,)),
                    f"{key} · outage; recovery not recorded",
                    "outage",
                )
            )
        self.events.sort(key=lambda event: event.tick)

    def filtered(self, owner=None, lane=None):
        return [
            e
            for e in self.events
            if (not owner or owner in e.owners) and (not lane or e.lane == lane)
        ]

    def targets(self, row):
        groups = {}
        for owner, vehicle in row["state"].get("agvs", {}).items():
            target = vehicle.get("target")
            if not isinstance(target, dict) or target.get("port") not in self.ports:
                continue
            port = self.ports[target["port"]]
            cell = (port.cell.x, port.cell.y)
            groups.setdefault(cell, []).append(
                (owner, "drop-off" if vehicle.get("job") else "pickup", target["port"])
            )
        return groups

    def history(self, owner, tick, length=12):
        if not length:
            return []
        result = []
        for when in range(max(0, tick - length), tick + 1):
            vehicle = self.recording.row(when)["state"].get("agvs", {}).get(owner, {})
            cell = vehicle.get("cell")
            if cell is not None:
                result.append(tuple(cell))
        return result


def decision_rows(row, owner=None, provider=None, job=None):
    """Read current semantic choices; unrecorded scores remain unavailable."""
    result = []
    for raw in row.get("decisions", ()):
        decision, selected = raw["decision"], raw["action"]
        candidates = decision["candidates"]
        if (
            owner
            and not job
            and decision["owner"] != owner
            and not (
                decision["group"] == "central"
                and any(
                    owner in (c.get("machine"), c.get("vehicle"), c.get("destination"))
                    for c in candidates
                )
            )
        ):
            continue
        if job and not any(
            c.get("job") == job
            or c["identity"] == job
            or isinstance(c["action"], (list, tuple))
            and job in c["action"]
            for c in candidates
        ):
            continue
        result.append(
            {
                "owner": decision["owner"],
                "role": decision["group"],
                "tick": decision["tick"],
                "result_tick": row["tick"],
                "stage": decision["stage"],
                "score_kind": "Action scores not recorded",
                "choices": [
                    {
                        "label": c["identity"],
                        "legal": True,
                        "selected": c["identity"] == selected
                        or c["action"] == selected,
                        "score": None,
                    }
                    for c in candidates
                ],
            }
        )
    return result
