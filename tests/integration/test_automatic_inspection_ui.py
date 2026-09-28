"""Editor and renderer contracts for independent inspection and local disposal."""

import os
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from smartsom.config.drawing_state import DrawingJob, DrawingState
from smartsom.config.factory_design import load_factory_design_file
from smartsom.domain.factory_design import Cell, SlotDesign
from smartsom.studio import editing
from smartsom.studio.drawing_dialog import DrawingStateDialog
from smartsom.studio.window import StudioWindow

pytestmark = pytest.mark.studio


def test_inspection_edit_undo_save_and_independent_frame_render(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = StudioWindow(data_dir=tmp_path / "user")
    try:
        doc = w.new_blank()
        w.editor.set_mode(True)
        design, sid = editing.create_resource(doc.design, "inspection_station", 2, 2)
        design, bid = editing.create_resource(design, "scrap_bin", 3, 2)
        station = replace(
            design.inspection_stations[0],
            slots=(SlotDesign("slot_001", Cell(0, 0), 4),),
        )
        design = editing.replace_resources(design, {sid: station})
        w.editor.commit(design, "Add inspection group", selection=(sid,))
        w.editor.set_mode(True)
        field = w.editor.properties.inputs["auto_disposal_bin_id"]
        field.control.setCurrentIndex(field.control.findData(bid))
        assert w.editor.apply_properties()
        assert doc.design.inspection_stations[0].auto_disposal_bin_id == bid
        doc.undo_stack.undo()
        assert doc.design.inspection_stations[0].auto_disposal_bin_id is None
        doc.undo_stack.redo()
        assert doc.design.inspection_stations[0].auto_disposal_bin_id == bid
        drawing = DrawingState(
            jobs=(
                DrawingJob(
                    order=1,
                    owner=sid,
                    slot="slot_001",
                    inspection_status="INSPECTING",
                    inspection_remaining=2,
                ),
                DrawingJob(
                    order=2,
                    owner=sid,
                    slot="slot_001",
                    inspection_status="INSPECTING",
                    inspection_remaining=1,
                ),
                DrawingJob(order=3, owner=sid, slot="slot_001", quality="PASS"),
                DrawingJob(
                    order=4,
                    owner=sid,
                    slot="slot_001",
                    quality="FAIL",
                    inspection_status="DISPOSING",
                    inspection_remaining=1,
                ),
            )
        )
        w.editor.commit(
            doc.design,
            "Job phases",
            selection=(sid,),
            authoring=doc.authoring.model_copy(update={"drawing_state": drawing}),
        )
        dialog = DrawingStateDialog(doc)
        assert dialog.read().jobs == drawing.jobs
        dialog.close()
        layer = doc.scene.drawing_layer
        points = {(p.x(), p.y()) for p, _, _ in layer.job_locations().values()}
        assert len(points) == 4
        assert (
            layer.state["stations"][sid]["jobs"]["d000004_a0"]["status"] == "DISPOSING"
        )
        path = tmp_path / "inspection.yaml"
        w.editor.save_to(doc, path)
        saved = load_factory_design_file(path)[0]
        assert saved.factory == doc.design
        assert saved.authoring.drawing_state.jobs == drawing.jobs
        # A recorded transfer animates from the inspection place to the local bin.
        before = layer.row
        import copy

        after = copy.deepcopy(before)
        after["state"]["storage"][sid]["slot_001"].remove("d000004_a0")
        after["events"] = [
            {
                "kind": "automatic_disposal",
                "job": "d000004_a0",
                "station": sid,
                "owner": bid,
            }
        ]
        layer.transition(after, 0.5)
        assert len(layer.paths()["d000004_a0"]) == 2
        layer.set_row(before)
        assert not layer.paths()  # Seeking restores the integer frame.
        app.processEvents()
    finally:
        for document in w.documents:
            document.undo_stack.setClean()
        w.editor.properties.pending = False
        w.close()
