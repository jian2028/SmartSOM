"""Adapt manually authored integer-frame annotations to the shared renderer."""

from PySide6.QtCore import QPointF

from smartsom.config.drawing_state import DrawingBuffer, DrawingMachine, DrawingState
from smartsom.studio.state_layer import FactoryStateLayer
from smartsom.studio.symbols import CELL_SIZE


def locations(design):
    result = [(m.machine_id, "") for m in design.machines]
    result += [(a.agv_id, "") for a in design.agvs]
    for b in design.buffers:
        result += (
            [(b.buffer_id, slot.slot_id) for slot in b.storage.slots]
            if b.storage.mode == "slots"
            else [(b.buffer_id, "pool")]
        )
    for station in design.inspection_stations:
        result += [(station.inspection_station_id, s.slot_id) for s in station.slots]
    return result


def validate_drawing(design, drawing):
    valid = set(locations(design))
    owners = {m.machine_id for m in design.machines} | {a.agv_id for a in design.agvs}
    occupied = set()
    for job in drawing.jobs:
        location = job.owner, job.slot
        if location not in valid:
            raise ValueError(f"Order {job.order}: choose an existing owner and slot")
        if job.owner in owners and job.owner in occupied:
            raise ValueError(f"{job.owner} can hold only one job")
        occupied.add(job.owner)
    for key, agv in drawing.agvs.items():
        if agv.conflict_with is not None and (
            agv.conflict_with == key
            or agv.conflict_with not in {a.agv_id for a in design.agvs}
        ):
            raise ValueError(f"{key}: choose another existing AGV for the conflict")
        if not (agv.x < design.grid.width and agv.y < design.grid.height):
            raise ValueError(f"{key}: drawing position must be inside the grid")


class DrawingEvidence:
    """Explicit drawing values; no statistical inference or future events."""

    def __init__(self, drawing):
        self.drawing = drawing

    def input_waiting(self, owner, state):
        return self.drawing.buffers.get(owner, DrawingBuffer()).waiting

    def input_progress(self, owner, tick):
        b = self.drawing.buffers.get(owner, DrawingBuffer())
        return b.total, b.remaining, 1

    def output_metrics(self, owner, tick, window):
        b = self.drawing.buffers.get(owner, DrawingBuffer())
        return b.delivered, b.passing_percent / 100, b.throughput


def attach_drawing(scene, drawing=None):
    drawing = drawing or DrawingState()
    state = {
        "machines": {},
        "agvs": {},
        "storage": {},
        "stations": {},
        "jobs": {},
        "metrics": {},
        "completed": [],
    }
    for m in scene.design.machines:
        value = drawing.machines.get(m.machine_id, DrawingMachine())
        state["machines"][m.machine_id] = {
            "job": None,
            "status": value.status,
            "mode": value.mode,
            "remaining": value.remaining,
            "elapsed": value.total - value.remaining,
            "down": value.status == "DOWN",
        }
    for agv in scene.design.agvs:
        value = drawing.agvs.get(agv.agv_id)
        state["agvs"][agv.agv_id] = {"job": None}
        if value:
            scene.entity_items[agv.agv_id].setPos(QPointF(value.x, value.y) * CELL_SIZE)
    for owner, slot in locations(scene.design):
        if owner not in state["machines"] and owner not in state["agvs"]:
            state["storage"].setdefault(owner, {})[slot] = []
    valid = set(locations(scene.design))
    for job in drawing.jobs:
        if (job.owner, job.slot) not in valid:
            continue  # Deleted geometry does not leave orphan symbols on the canvas.
        jid = job.identifier
        state["jobs"][jid] = {"quality": job.quality}
        if job.owner in state["machines"]:
            state["machines"][job.owner]["job"] = jid
        elif job.owner in state["agvs"]:
            state["agvs"][job.owner]["job"] = jid
            scene.entity_items[job.owner].set_loaded(True)
        else:
            state["storage"][job.owner][job.slot].append(jid)
    for station in scene.design.inspection_stations:
        key = station.inspection_station_id
        value = drawing.stations.get(key)
        state["stations"][key] = {
            "batch": [
                j for slot in state["storage"].get(key, {}).values() for j in slot
            ]
            if value and value.inspecting
            else [],
            "total": value.total if value else 0,
            "remaining": value.remaining if value else 0,
        }
    state["metrics"] = {f"scrap:{key}": n for key, n in drawing.disposed.items()}
    conflicts = [key for key, value in drawing.agvs.items() if value.conflict]
    layer = FactoryStateLayer(scene.design, scene.entity_items)
    layer.evidence = DrawingEvidence(drawing)
    layer.charging_label = "CHARGING"
    layer.charging_preview = {
        key for key, value in drawing.agvs.items() if value.charging
    }
    layer.set_row(
        {
            "tick": drawing.tick,
            "state": state,
            "actions": {"agvs": [(key, "UP") for key in conflicts]},
            "rejections": {f"agv:{key}": "conflict" for key in conflicts},
        }
    )
    scene.addItem(layer)
    scene.drawing_layer = layer
    return layer


def reconcile_drawing(before, after, drawing, renames=None):
    """Keep annotations attached through geometry edits and semantic ID renames."""
    renames = renames or {}
    data = drawing.model_dump()
    groups = {
        "machines": {m.machine_id for m in after.machines},
        "agvs": {a.agv_id for a in after.agvs},
        "buffers": {b.buffer_id for b in after.buffers},
        "stations": {s.inspection_station_id for s in after.inspection_stations},
        "disposed": {s.scrap_bin_id for s in after.scrap_bins},
    }
    for name, keys in groups.items():
        data[name] = {
            renames.get(k, k): v
            for k, v in data[name].items()
            if renames.get(k, k) in keys
        }
    valid = set(locations(after))
    jobs = []
    for job in drawing.jobs:
        owner = renames.get(job.owner, job.owner)
        slot = renames.get(job.slot, job.slot)
        if (owner, slot) in valid:
            jobs.append(job.model_copy(update={"owner": owner, "slot": slot}))
    data["jobs"] = tuple(jobs)
    old_agvs = {renames.get(a.agv_id, a.agv_id): a for a in before.agvs}
    for a in after.agvs:
        if a.agv_id in data["agvs"]:
            previous = old_agvs.get(a.agv_id)
            v = data["agvs"][a.agv_id]
            if previous and previous.initial_cell != a.initial_cell:
                v["x"], v["y"] = a.initial_cell.x, a.initial_cell.y
            partner = renames.get(v.get("conflict_with"), v.get("conflict_with"))
            v["conflict_with"] = partner if partner in groups["agvs"] else None
            v["x"] = min(v["x"], after.grid.width - 1)
            v["y"] = min(v["y"], after.grid.height - 1)
    return DrawingState.model_validate(data)
