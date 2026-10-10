"""Optional recording closes resources and avoids repeated restored steps."""

import json
import random
import sys
from types import SimpleNamespace

import pytest

from smartsom.config.codec import ConfigurationError
from smartsom.telemetry.tensorboard import TensorBoardRecorder


def test_disabled_does_not_load_writer_or_create_directory(tmp_path):
    recorder = TensorBoardRecorder(tmp_path, False)
    recorder.record({"value": 1}, 2)
    recorder.close()
    assert not (tmp_path / "logs").exists()


def test_enabled_requires_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smartsom.telemetry.tensorboard.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(ConfigurationError, match="--extra tensorboard"):
        TensorBoardRecorder(tmp_path, True)
    assert not (tmp_path / "logs").exists()


def test_resume_deduplicates_each_tag_and_closes_writer(tmp_path, monkeypatch):
    writes, writers = [], []

    class Writer:
        def __init__(self, path):
            __import__("pathlib").Path(path).mkdir(parents=True, exist_ok=True)
            self.closed = False
            self.flushed = False
            writers.append(self)
            random.random()

        def add_scalar(self, tag, value, global_step):
            writes.append((tag, value, global_step))
            random.random()

        def flush(self):
            self.flushed = True

        def close(self):
            self.closed = True
            random.random()

    monkeypatch.setitem(
        sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=Writer)
    )
    monkeypatch.setattr(
        "smartsom.telemetry.tensorboard.importlib.util.find_spec",
        lambda name: SimpleNamespace(),
    )
    state = random.getstate()
    first = TensorBoardRecorder(tmp_path, True)
    first.record(
        {"training/return": 4, "validation/return": 8, "nan": float("nan")}, 10
    )
    first.close()
    second = TensorBoardRecorder(tmp_path, True)
    second.record({"training/return": 99, "validation/new": 3}, 10)
    second.record({"training/return": 5}, 11)
    second.close()
    assert writes == [
        ("training/return", 4.0, 10),
        ("validation/return", 8.0, 10),
        ("validation/new", 3.0, 10),
        ("training/return", 5.0, 11),
    ]
    assert all(w.closed and w.flushed for w in writers)
    assert random.getstate() == state
    assert (
        json.loads((tmp_path / "logs/tensorboard/recorded-steps.json").read_text())[
            "training/return"
        ]
        == 11
    )
