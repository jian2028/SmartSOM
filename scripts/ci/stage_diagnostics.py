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
    "driver-stderr.log",
    "worker-error.json",
    "tuning-runtime.json",
    "owner.json",
    "observer-summary.json",
    "admission-diagnostics.json",
    "deadline-summary.json",
}


def failed_directories(source, results):
    """Accept only failed-test fixture paths confined to this suite's temp root."""
    source = Path(source).resolve()
    directories = set()
    if results is None:
        return directories
    for report in Path(results).rglob("*.json"):
        if report.is_symlink() or report.stat().st_size > 8388608:
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            paths = data.get("failed_test_paths", {}).values()
            for value in paths:
                path = Path(value)
                if path.is_symlink():
                    continue
                path = path.resolve()
                if path.is_relative_to(source) and path.is_dir():
                    directories.add(path)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return directories


def stage(
    source,
    destination,
    *,
    max_files=128,
    max_file_bytes=262144,
    max_bytes=8388608,
    max_entries=20000,
    results=None,
):
    source_is_link = Path(source).is_symlink()
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {"files": [], "bytes": 0, "scanned": 0, "limited": False, "tails": {}}
    failed = set() if source_is_link else failed_directories(source, results)

    def is_failed(path):
        return any(path.resolve().is_relative_to(directory) for directory in failed)

    candidates = []
    directories = [source] if source.is_dir() and not source_is_link else []
    directories.extend(sorted(failed, reverse=True))
    visited = set()
    while directories:
        directory = directories.pop()
        if directory in visited:
            continue
        visited.add(directory)
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
                if (
                    name not in NAMES
                    and not (
                        name in {"stdout.log", "stderr.log", "progress.json"}
                        and is_failed(Path(entry.path))
                    )
                    and not (name.startswith("evaluation-") and name.endswith(".log"))
                ):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                candidates.append(Path(entry.path))
    # Failure logs/summaries precede other failed-fixture files, then first-admission
    # evidence from other fixtures. Scan/byte/no-symlink bounds remain unchanged.
    failure_names = {
        "stderr.log": 0,
        "stdout.log": 0,
        "observer-summary.json": 1,
        "deadline-summary.json": 1,
        "progress.json": 2,
        "admission-diagnostics.json": 3,
    }
    candidates.sort(
        key=lambda path: (
            not is_failed(path),
            failure_names.get(path.name, 4)
            if is_failed(path)
            else int(path.name != "admission-diagnostics.json"),
            str(path),
        )
    )
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        if len(result["files"]) >= max_files:
            result["limited"] = True
            continue
        descriptor = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                continue
            if (
                info.st_size > max_file_bytes
                and candidate.name not in {"stdout.log", "stderr.log"}
            ) or result["bytes"] + min(info.st_size, max_file_bytes) > max_bytes:
                result["limited"] = True
                continue
            offset = max(0, info.st_size - max_file_bytes)
            stream.seek(offset)
            payload = stream.read(max_file_bytes + 1)
        if len(payload) > max_file_bytes or result["bytes"] + len(payload) > max_bytes:
            result["limited"] = True
            continue
        relative = Path(candidate).relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        result["files"].append(str(relative))
        if offset:
            result["tails"][str(relative)] = {
                "original_bytes": info.st_size,
                "offset": offset,
            }
            result["limited"] = True
        result["bytes"] += len(payload)
    (destination / "staging-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            stage(arguments.source, arguments.destination, results=arguments.results)
        )
    )
