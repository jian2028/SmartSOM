"""Create a separate stage-zero batch without editing any frozen experiment.

Run with the legacy runtime's locked environment. The originals remain the
scientific source; only queue metadata and equivalent absolute paths change.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def prepare(source, destination):
    originals = sorted(source.glob("stage[23]-long-*/experiment.yaml"))
    if len(originals) != 10:
        raise ValueError("expected the ten stage 2/3 long independent arms")
    prepared = []
    for path in originals:
        before = yaml.safe_load(path.read_text())
        if before["train_episodes"] != 1000 or before["train_seeds"] != [101]:
            raise ValueError(f"unexpected training budget/seed: {path}")
        after = copy.deepcopy(before)
        after["batch"] = {"stage": 0, "parallel_files": 8}
        for key in ("algorithm", "scenario"):
            after[key] = str((path.parent / before[key]).resolve(strict=True))
        after["case_sets"] = {
            key: str((path.parent / value).resolve(strict=True))
            for key, value in before["case_sets"].items()
        }
        target = destination / ("rolling-" + path.parent.name) / "experiment.yaml"
        content = yaml.safe_dump(after, sort_keys=False, allow_unicode=True)
        if target.exists() and target.read_text() != content:
            raise FileExistsError(f"refusing to change existing entry: {target}")
        prepared.append((target, content, path))
    for target, content, _ in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return [
        {
            "entry": target.parent.name,
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generated_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        for target, content, path in prepared
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source.resolve(), args.destination.resolve()),
            indent=2,
            ensure_ascii=False,
        )
    )
