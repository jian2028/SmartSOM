"""Small archived-source recording for current read-only viewer contracts."""

import shutil
from pathlib import Path

from smartsom.config.scenario_snapshot import scenario_from_snapshot
from smartsom.trace.production import Playback


def recording(root):
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "fixtures/historical-hand", root
    )
    return root, scenario_from_snapshot(Playback(root).manifest["inputs"]["scenario"])
