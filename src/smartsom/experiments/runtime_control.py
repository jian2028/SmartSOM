"""Read-only snapshots of current Tune observations and historical controls.

Online queue mutation, task controls and resource leases are retired. Historical
journals remain readable without applying or acknowledging their old commands.
"""

import json
from pathlib import Path


def snapshot(root):
    """Read an atomic observation without creating files or changing commands."""
    path = Path(root).expanduser().resolve() / "control/runtime.json"
    return json.loads(path.read_text()) if path.exists() else {}
