"""CLI discovery and framework isolation do not require a window or Qt."""

import os
import subprocess
import sys
from pathlib import Path


def run_script(source):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    return subprocess.run(
        [sys.executable, "-c", source], env=env, capture_output=True, text=True
    )


def test_design_and_studio_namespace_import_without_optional_frameworks():
    result = run_script("""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6','ray','torch','gymnasium','pettingzoo'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
import smartsom
import smartsom.studio
from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import FactoryDesign
assert 'PySide6' not in sys.modules
""")
    assert result.returncode == 0, result.stderr


def test_cli_help_and_missing_dependency_are_actionable():
    prefix = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'PySide6':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from smartsom.experiments.cli import main
"""
    for args in ("['--help']", "['studio', '--help']"):
        result = run_script(prefix + f"\nmain({args})")
        assert result.returncode == 0, result.stderr
        assert "studio" in result.stdout
    result = run_script(prefix + "\nsys.exit(main(['studio']))")
    assert result.returncode == 2
    assert "uv sync --extra studio" in result.stderr


def test_cli_dispatch_passes_paths_without_entering_experiments():
    result = run_script("""
import sys, types
from pathlib import Path
from smartsom.experiments.cli import main
module = types.ModuleType('smartsom.studio.app')
def launch(paths):
    assert paths == [Path('one.yaml'), Path('two.yml')]
    return 17
module.main = launch
sys.modules['smartsom.studio.app'] = module
assert main(['studio', 'one.yaml', 'two.yml']) == 17
""")
    assert result.returncode == 0, result.stderr


def test_public_historical_playback_dispatch_and_seed_ambiguity(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from smartsom.experiments.cli import main

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "smartsom.studio.historical_playback",
        SimpleNamespace(
            historical_playback_window=lambda root, seed: (
                calls.append((root, seed)) or 0
            )
        ),
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"schema": "smartsom.spatial-run/v1"})
    )
    (tmp_path / "frames-24004.jsonl").write_text("")
    assert main(["playback", str(tmp_path)]) == 0
    assert calls == [(tmp_path, 24004)]
    (tmp_path / "frames-24005.jsonl").write_text("")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(["playback", str(tmp_path)]) == 2
    assert main(["playback", str(tmp_path), "--seed", "24005"]) == 0
    assert calls[-1] == (tmp_path, 24005)
    assert all(p.read_bytes() == raw for p, raw in before.items())
