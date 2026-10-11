"""Stage bounded regular diagnostics without following checkpoint symlinks."""

import argparse
import json
import os
import stat
from pathlib import Path

NAMES = {
    "calibration.json",
    "batch.json",
    "probe.log",
    "study-process.log",
    "report.json",
    "summary.json",
}


def stage(
    source,
    destination,
    *,
    max_files=128,
    max_file_bytes=262144,
    max_bytes=8388608,
    max_entries=20000,
):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {"files": [], "bytes": 0, "scanned": 0, "limited": False}
    directories = [source] if source.is_dir() and not source.is_symlink() else []
    while directories:
        directory = directories.pop()
        if directory.is_symlink():
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                if result["scanned"] >= max_entries:
                    result["limited"] = True
                    directories.clear()
                    break
                result["scanned"] += 1
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories.append(Path(entry.path))
                    continue
                name = entry.name
                if name not in NAMES and not (
                    name.startswith("evaluation-") and name.endswith(".log")
                ):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if len(result["files"]) >= max_files:
                    result["limited"] = True
                    continue
                descriptor = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    if (
                        info.st_size > max_file_bytes
                        or result["bytes"] + info.st_size > max_bytes
                    ):
                        result["limited"] = True
                        continue
                    payload = stream.read(max_file_bytes + 1)
                if (
                    len(payload) > max_file_bytes
                    or result["bytes"] + len(payload) > max_bytes
                ):
                    result["limited"] = True
                    continue
                relative = Path(entry.path).relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                result["files"].append(str(relative))
                result["bytes"] += len(payload)
    (destination / "staging-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(stage(arguments.source, arguments.destination)))
