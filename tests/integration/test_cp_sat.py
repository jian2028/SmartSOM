"""CP-SAT has no grid adapter: every public entry rejects it before allocation.

The former matrix optimum/replay checks belong to their historical checkout.
Raw benchmark references remain unchanged and independently checked in unit tests.
No historical schedule can establish an optimum for cell movement and capacity.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cp


@pytest.mark.parametrize("name", ["ft06", "pyjobshop_fjsp", "mk01"])
@pytest.mark.parametrize("command", ["validate", "run"])
def test_grid_cli_rejects_cp_sat_before_any_solver_or_run_artifact(
    tmp_path, name, command
):
    source = ROOT / f"configs/runs/{name}_cp.yaml"
    document = yaml.safe_load(source.read_text())
    for field in ("scenario", "algorithm"):
        document[field] = str((source.parent / document[field]).resolve())
    output = tmp_path / "runs"
    document["output_root"] = str(output)
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(document))
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'pyjobshop', 'ortools'}:
        raise AssertionError('unsupported grid CP imported a solver')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from smartsom.experiments.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, command, "--config", str(config)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "CP-SAT" in result.stderr and "grid" in result.stderr
    assert "AssertionError" not in result.stderr
    assert not output.exists()
