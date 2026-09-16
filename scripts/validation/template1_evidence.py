"""Freeze the dirty development source without committing generated evidence."""

import hashlib
import subprocess
import zipfile
from pathlib import Path

from smartsom.trace.production import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def capture_source(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path("pyproject.toml"), Path("uv.lock")]
    for folder in ("src", "configs", "tests", "scripts/validation"):
        paths.extend(
            p.relative_to(ROOT)
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yaml", ".json")
        )
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(ROOT / path, str(path))
    atomic_json(
        output / "hashes.json",
        {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    )
    for name, args in [
        ("tracked.diff", ["diff", "HEAD"]),
        ("git-status.txt", ["status", "--short", "--branch"]),
        ("head.txt", ["rev-parse", "HEAD"]),
    ]:
        (output / name).write_bytes(subprocess.check_output(["git", *args], cwd=ROOT))
    return str(output)
