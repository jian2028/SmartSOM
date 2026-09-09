"""Persist pytest's collected nodeids for the fixed acceptance coverage gate."""

import json
import os
from pathlib import Path


def pytest_collection_finish(session):
    destination = os.environ.get("SMARTSOM_FEATURE_COLLECTION")
    if destination:
        Path(destination).write_text(
            json.dumps([item.nodeid for item in session.items], indent=2) + "\n",
            encoding="utf-8",
        )
