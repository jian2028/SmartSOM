"""Embed a trace into the standalone viewer to produce a single shareable HTML file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VIEWER_TEMPLATE = Path(__file__).with_name("viewer.html")
PLACEHOLDER = "__REPLAY_DATA__"


def render_html(
    manifest: dict[str, Any], frames: list[dict[str, Any]], out_path: Path | str
) -> Path:
    template = VIEWER_TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise RuntimeError(f"{VIEWER_TEMPLATE} must contain {PLACEHOLDER} exactly once")
    payload = json.dumps(
        {"manifest": manifest, "frames": frames}, separators=(",", ":")
    )
    # Keep the JSON from closing the surrounding <script> tag.
    payload = payload.replace("</", "<\\/")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    return out_path
