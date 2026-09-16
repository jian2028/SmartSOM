"""Restore the frozen September 14 comparison, without changing the main runtime."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/historical-replay-24004"
REFERENCE = Path("artifacts/spatial/memory-demo-0914")
CASES = {
    "rule": "rule/rule-24004",
    "marl": "starting_actor/checkpoint-00000256-24004",
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(old):
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing evidence: {OUTPUT}")
    source = old / REFERENCE / "reference-dev-source/execution"
    manifest = read(source / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Historical source hash mismatch: {name}")
    shutil.copytree(source, OUTPUT / "execution")
    for case, relative in CASES.items():
        original = old / REFERENCE / "reference-dev" / relative
        target = OUTPUT / "original" / case
        shutil.copytree(original, target)
    original_model = read(OUTPUT / "original/marl/run.json")
    checkpoint = Path(original_model["checkpoint"])
    if sha(checkpoint) != original_model["checkpoint_sha256"]:
        raise ValueError("Historical model hash mismatch")
    shutil.copy2(checkpoint, OUTPUT / "checkpoint.pt")
    write(
        OUTPUT / "provenance.json",
        {
            "historical_root": str(old),
            "historical_source_manifest": manifest["sha256"],
            "checkpoint_sha256": sha(OUTPUT / "checkpoint.pt"),
            "main_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "description": "Reconstruction from frozen source, not an original recording",
            "seed": 24004,
        },
    )
    (OUTPUT / "main-workspace.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    )
    for case in CASES:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "record", case],
            check=True,
        )


def runtime():
    sys.path.insert(0, str(OUTPUT / "execution/src"))


def record(case):
    runtime()
    from smartsom.spatial.api import SpatialExperimentConfig, evaluate, run
    from smartsom.spatial.models import SpatialConfig

    original = OUTPUT / "original" / case
    expected = read(original / "run.json")
    config = SpatialExperimentConfig(
        environment=SpatialConfig(**read(original / "recipe.json")["environment"]),
        factory_path=str(original / "factory.yaml"),
        output_dir=str(OUTPUT / case),
        seeds=(24004,),
        record_frames=True,
        record_trace=True,
        deterministic=expected.get("deterministic", True),
    )
    result = (
        run(config) if case == "rule" else evaluate(OUTPUT / "checkpoint.pt", config)
    )
    actual = read(result.run_dir / "run.json")
    old_episode = read(original / "episode-24004.json")
    episode = read(result.run_dir / "episode-24004.json")
    differences = [
        key
        for key in old_episode.keys() | episode.keys()
        if old_episode.get(key) != episode.get(key)
    ]
    report = {
        "case": case,
        "exact_episode_match": not differences,
        "different_fields": differences,
        "factory_match": actual["factory_sha256"] == expected["factory_sha256"],
        "spatial_source_match": actual["spatial_source_sha256"]
        == expected["spatial_source_sha256"],
        "fulfilled": episode["fulfilled"],
        "tick": episode["tick"],
        "recording": str(result.run_dir),
    }
    report["passed"] = all(
        report[k]
        for k in ("exact_episode_match", "factory_match", "spatial_source_match")
    )
    write(OUTPUT / f"{case}-verification.json", report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Historical reconstruction differs; inspect verification")


def play(case):
    if not read(OUTPUT / f"{case}-verification.json")["passed"]:
        raise ValueError("Reconstruction has not passed verification")
    runtime()
    from smartsom.spatial.cli import execute

    execute(argparse.Namespace(action="replay", run_dir=OUTPUT / case, seed=24004))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "record", "play"))
    parser.add_argument("case", choices=tuple(CASES), nargs="?")
    parser.add_argument(
        "--historical-root",
        type=Path,
        help="Frozen historical source root (prepare only)",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.historical_root is None:
            parser.error(
                "prepare requires --historical-root PATH to the frozen evidence"
            )
        prepare(args.historical_root)
    elif args.case is None:
        parser.error("record/play requires rule or marl")
    else:
        required = OUTPUT / (
            "execution" if args.action == "record" else f"{args.case}-verification.json"
        )
        if not required.exists():
            parser.error(
                "Historical evidence is not included in a clone. Obtain the evidence "
                "bundle described in docs/historical-replay.md; for a standalone demo "
                "see docs/template1-replay.md."
            )
        if args.action == "record":
            record(args.case)
        else:
            play(args.case)


if __name__ == "__main__":
    main()
