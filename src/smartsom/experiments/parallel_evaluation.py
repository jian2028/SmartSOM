"""Bounded native case workers; the caller alone owns ordered durable results."""

import multiprocessing
import traceback
from multiprocessing.connection import wait
from pathlib import Path

from smartsom.experiments import control
from smartsom.experiments.resource_guard import ResourceGuardExceeded
from smartsom.experiments.resource_guard import scope as resource_scope


def _worker(connection, cancelled, settings):
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.experiments.reservation_runtime import evaluate_case, identity

    def boundary():
        if cancelled.is_set():
            raise control.StopRequested("evaluation worker cancelled")
        control.boundary(settings["output"])

    try:
        prepared = load_inputs(settings["experiment"])
        if identity(prepared[-1]) != settings["inputs"]:
            raise ValueError("evaluation inputs changed before worker startup")
        with resource_scope(
            Path(settings["output"])
            / "workers"
            / str(multiprocessing.current_process().pid),
            prepared[0].resources.guard,
        ):
            while not cancelled.is_set():
                if not connection.poll(0.1):
                    continue
                task = connection.recv()
                if task is None:
                    return
                boundary()
                result = evaluate_case(
                    settings,
                    task,
                    prepared,
                    control_check=boundary,
                    on_progress=lambda row: connection.send(
                        ("progress", task["index"], row)
                    ),
                )
                connection.send(("result", task["index"], result))
    except BaseException as exc:
        try:
            # The driver retry contract uses isinstance, including subclasses
            # such as Pydantic ValidationError, not exact exception names.
            category = next(
                (
                    kind.__name__
                    for kind in (
                        control.PauseRequested,
                        control.StopRequested,
                        KeyboardInterrupt,
                        SystemExit,
                        ResourceGuardExceeded,
                        ValueError,
                        AssertionError,
                        TimeoutError,
                    )
                    if isinstance(exc, kind)
                ),
                "RuntimeError",
            )
            connection.send(("error", category, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def case_results(settings, tasks, prepared, *, workers, on_progress):
    """Yield in frozen order, bounding active and buffered cases by worker count.

    Every task constructs/seeds/loads its own policy, including on reused workers.
    Generator close cooperatively cancels and joins all children before releasing
    the caller's stage lock. No child writes receipts, summaries or tracking.
    """
    from smartsom.experiments.reservation_runtime import evaluate_case

    if workers == 1:
        for task in tasks:
            yield (
                task["index"],
                evaluate_case(
                    settings,
                    task,
                    prepared,
                    on_progress=lambda row, i=task["index"]: on_progress(i, row),
                ),
            )
        return
    if not tasks:
        return
    if any(not spec.target for _, spec in prepared[3].component_items()):
        raise ValueError("parallel evaluation supports built-in components only")
    context = multiprocessing.get_context("spawn")
    cancelled = context.Event()
    children = {}
    active = {}
    completed = {}
    pending = iter(tasks)
    try:
        for _ in range(min(workers, len(tasks))):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, cancelled, settings))
            try:
                process.start()
            except BaseException:
                child.close()
                if process.pid is not None:
                    children[parent] = process
                else:
                    parent.close()
                raise
            child.close()
            children[parent] = process
            owner = control.CURRENT.get()
            if owner is not None:
                owner.capture()
            task = next(pending)
            active[parent] = task["index"]
            parent.send(task)
        for expected in tasks:
            control.boundary(settings["output"])
            index = expected["index"]
            while index not in completed:
                control.boundary(settings["output"])
                connections = list(active)
                ready = wait(
                    connections + [children[c].sentinel for c in connections],
                    timeout=0.1,
                )
                for conn in connections:
                    if conn in ready:
                        try:
                            kind, item, value = conn.recv()
                        except EOFError as exc:
                            raise RuntimeError(
                                "evaluation worker exited without a result"
                            ) from exc
                        if kind == "error":
                            if item == "PauseRequested":
                                raise control.PauseRequested(value)
                            if item in {"StopRequested", "KeyboardInterrupt"}:
                                raise control.StopRequested(value)
                            exception = {
                                "ValueError": ValueError,
                                "AssertionError": AssertionError,
                                "ResourceGuardExceeded": ResourceGuardExceeded,
                                "TimeoutError": TimeoutError,
                                "SystemExit": SystemExit,
                            }.get(item, RuntimeError)
                            raise exception("evaluation worker failed:\n" + value)
                        if item != active[conn]:
                            raise RuntimeError(
                                "evaluation worker returned another case identity"
                            )
                        if kind == "progress":
                            on_progress(item, value)
                        elif kind == "result":
                            completed[item] = (conn, value)
                            del active[conn]
                        else:
                            raise RuntimeError("invalid evaluation worker response")
                    elif children[conn].sentinel in ready:
                        raise RuntimeError("evaluation worker exited without a result")
            conn, value = completed.pop(index)
            control.boundary(settings["output"])
            yield index, value
            control.boundary(settings["output"])
            task = next(pending, None)
            if task is not None:
                active[conn] = task["index"]
                conn.send(task)
    finally:
        cancelled.set()
        # Drain IPC while joining: a child may already be sending a large report.
        while any(process.is_alive() for process in children.values()):
            for conn in list(children):
                try:
                    if conn.poll(0.02):
                        conn.recv()
                except (EOFError, OSError):
                    pass
            for process in children.values():
                process.join(timeout=0.02)
        for conn, process in children.items():
            conn.close()
            process.join()
            process.close()
