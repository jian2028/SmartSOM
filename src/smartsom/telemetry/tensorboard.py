"""Optional event sink; serving is an explicit command, never a worker side effect."""

import subprocess
import sys
import time
from pathlib import Path


class TensorBoardRecorder:
    def __init__(self, directory):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self.writer = EventFileWriter(str(Path(directory)))

    def log(self, values, step):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self.writer.add_event(
            Event(
                wall_time=time.time(),
                step=step,
                summary=Summary(
                    value=[
                        Summary.Value(tag=key, simple_value=value)
                        for key, value in values.items()
                    ]
                ),
            )
        )
        self.writer.flush()

    def close(self):
        self.writer.close()


def serve(root, port=6006):
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    root = Path(root).resolve()
    if not any(root.rglob("events.out.tfevents.*")):
        raise ValueError(
            "no TensorBoard events in this run; enable --tensorboard when launching"
        )
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )
