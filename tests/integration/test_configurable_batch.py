"""Disposable public directory execution of declarative PPO/DQN contracts."""

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from test_author_batch_cli_learning import _alive, _command, _remember_children, _wait
from test_author_learning import ROOT, _inputs, _require_cpu, _write

pytestmark = pytest.mark.learning


class _ResumeDeadline:
    """Monotonic watchdog for this fixed scientific acceptance fixture.

    Heartbeats/status changes/repeated counters cannot extend either deadline.
    Validation tick maxima belong to separate rounds; inactive stale ticks never
    seed the next round. The fixture's unchanged budgets bound every counter.
    """

    def __init__(self, entry_id, *, started, total, stalled, restored=None):
        self.entry_id = entry_id
        self.started = self.last_progress = started
        self.total, self.stalled = total, stalled
        self.maxima = {}
        self.reason = None
        self.observe(
            {
                "schema": "smartsom.runtime-progress/v1",
                "tasks": [{"id": entry_id, "values": restored or {}}],
            },
            started,
        )

    def observe(self, snapshot, now):
        if now - self.started >= self.total:
            self.reason = "hard total deadline"
        elif now - self.last_progress >= self.stalled:
            self.reason = "no semantic progress deadline"
        if self.reason is not None:
            return False
        if snapshot.get("schema") != "smartsom.runtime-progress/v1":
            return True
        for task in snapshot.get("tasks", []):
            if task.get("id") != self.entry_id:
                continue
            values = task.get("values", {})
            limits = {
                "physical_ticks": 512,
                "updates": 8,
                "validation_batches_finished": 3,
                "evaluation_finished": 1,
            }
            counters = {key: values.get(key) for key in limits}
            round_number = values.get("validation_round")
            if (
                type(round_number) is int
                and 1 <= round_number <= 3
                and values.get("validation_case_active") is True
            ):
                key = f"validation_tick_round_{round_number}"
                counters[key], limits[key] = values.get("validation_tick"), 512
            if values.get("evaluation_case_active") is True:
                counters["evaluation_tick"] = values.get("evaluation_tick")
                limits["evaluation_tick"] = 512
            for key, value in counters.items():
                if type(value) is int and 0 <= value <= limits[key]:
                    if value > self.maxima.get(key, 0):
                        self.maxima[key] = value
                        self.last_progress = now
        return True


def _progress_snapshot(path):
    try:
        with path.open("rb") as stream:
            payload = stream.read(262145)
        if len(payload) > 262144:
            return {}
        result = json.loads(payload)
        if isinstance(result, dict) and isinstance(result.get("tasks", []), list):
            if all(
                isinstance(task, dict) and isinstance(task.get("values", {}), dict)
                for task in result.get("tasks", [])
            ):
                return result
    except (OSError, ValueError):
        pass
    return {}


def _public_result(
    directory,
    environment,
    arguments,
    *,
    timeout,
    progress_path=None,
    entry_id=None,
    stalled=None,
    restored=None,
):
    """Keep exact descendant identities until timeout cleanup finishes.

    subprocess.run kills only the CLI on timeout, leaving its native/Ray child
    driver running. Preserve the failure and release only this command's owned
    process tree before another case measures available resources.
    """
    with _command(
        directory, environment, *arguments, register_exit_children=True
    ) as command:
        process, owned, stdout, stderr = command
        watchdog = None
        if progress_path is not None:
            watchdog = _ResumeDeadline(
                entry_id,
                started=time.monotonic(),
                total=timeout,
                stalled=stalled,
                restored=restored,
            )
            # Baseline restored counters; they are not newly completed work.
            watchdog.observe(_progress_snapshot(progress_path), watchdog.started)
        next_observation = 0.0

        def completed():
            nonlocal next_observation
            now = time.monotonic()
            if watchdog is not None:
                snapshot = {}
                if now >= next_observation:
                    snapshot = _progress_snapshot(progress_path)
                    next_observation = now + 0.25
                if not watchdog.observe(snapshot, now):
                    pytest.fail(
                        f"public resume {watchdog.reason}:\n"
                        f"{stdout.read_text()}\n{stderr.read_text()}"
                    )
            return process.poll() is not None

        try:
            _wait(command, completed, timeout=timeout, allow_exit=True)
        finally:
            if watchdog is not None:
                (directory / "deadline-summary.json").write_text(
                    json.dumps(
                        {
                            "total_seconds": timeout,
                            "stalled_seconds": stalled,
                            "elapsed_seconds": time.monotonic() - watchdog.started,
                            "since_progress_seconds": time.monotonic()
                            - watchdog.last_progress,
                            "maxima": watchdog.maxima,
                            "reason": watchdog.reason,
                        }
                    )
                    + "\n"
                )
        if process.returncode == 0:
            _wait(command, lambda: not _alive(owned), timeout=10, allow_exit=True)
        return subprocess.CompletedProcess(
            arguments, process.returncode, stdout.read_text(), stderr.read_text()
        )


def test_public_timeout_cleans_exact_children_and_preserves_sibling(
    tmp_path, monkeypatch
):
    import psutil
    import test_author_batch_cli_learning as public

    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        f"Path({str(pid_file)!r}).write_text(str(child.pid));time.sleep(60)"
    )
    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        monkeypatch.setattr(public, "CLI", [sys.executable, "-c", script])
        with pytest.raises(pytest.fail.Exception, match="expected boundary"):
            _public_result(tmp_path / "timeout", os.environ.copy(), [], timeout=2)
        assert sibling.poll() is None
        pid = int(pid_file.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.parametrize("attempt", range(12))
def test_public_early_exit_registers_detached_child_and_preserves_sibling(
    tmp_path, monkeypatch, attempt
):
    import psutil
    import test_author_batch_cli_learning as public

    identity = tmp_path / "child.json"
    script = (
        "import subprocess,sys,json,psutil;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        f"Path({str(identity)!r}).write_text(json.dumps([child.pid,"
        "psutil.Process(child.pid).create_time()]));sys.exit(7)"
    )
    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        monkeypatch.setattr(public, "CLI", [sys.executable, "-c", script])
        if attempt % 2:
            launch = subprocess.Popen

            def delayed_observation(*args, **kwargs):
                process = launch(*args, **kwargs)
                process.wait(timeout=2)
                return process

            # Half the probes force the CLI to exit before the tracker obtains
            # even its first parent identity, rather than relying on timing.
            monkeypatch.setattr(public.subprocess, "Popen", delayed_observation)
        result = _public_result(
            tmp_path / f"early-exit-{attempt}", os.environ.copy(), [], timeout=2
        )
        assert result.returncode == 7
        assert sibling.poll() is None
        pid, born = json.loads(identity.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).create_time() != born
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.parametrize("depth", [2, 4, 8])
@pytest.mark.parametrize("fanout", [1, 4])
@pytest.mark.parametrize("observe_after_exit", [False, True])
@pytest.mark.parametrize("explicit_environment", [False, True])
def test_public_intermediate_exit_keeps_all_generations_registered(
    tmp_path, monkeypatch, depth, fanout, observe_after_exit, explicit_environment
):
    import psutil
    import test_author_batch_cli_learning as public

    program = tmp_path / "generations.py"
    program.write_text(
        "import subprocess,sys,json,psutil,time,os\nfrom pathlib import Path\n"
        f"directory=Path({str(tmp_path)!r})\n"
        "def main(level):\n"
        f"    environment={{'PATH':os.environ.get('PATH','')}} if {explicit_environment!r} else None\n"
        "    if level>1:\n"
        "        child=subprocess.Popen([sys.executable,__file__,str(level-1)],env=environment)\n"
        "        assert child.wait()==7;return 7\n"
        f"    for index in range({fanout}):\n"
        "        ready=directory/f'ready-{index}'\n"
        "        code=f'from pathlib import Path;import time;Path({str(ready)!r}).touch();time.sleep(60)'\n"
        "        child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,env=environment)\n"
        "        (directory/f'leaf-{index}.json').write_text(json.dumps([child.pid,psutil.Process(child.pid).create_time()]))\n"
        "    deadline=time.monotonic()+5\n"
        f"    while not all((directory/f'ready-{{index}}').exists() for index in range({fanout})):\n"
        "        assert time.monotonic()<deadline;time.sleep(.01)\n"
        "    return 7\n"
        "if __name__=='__main__':sys.exit(main(int(sys.argv[1])))\n"
    )

    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        monkeypatch.setattr(public, "CLI", [sys.executable, str(program), str(depth)])
        if observe_after_exit:
            launch = subprocess.Popen

            def delayed_observation(*args, **kwargs):
                process = launch(*args, **kwargs)
                process.wait(timeout=5)
                return process

            monkeypatch.setattr(public.subprocess, "Popen", delayed_observation)
        directory = tmp_path / "multigeneration"
        result = _public_result(directory, os.environ.copy(), [], timeout=5)
        assert result.returncode == 7, result.stderr
        assert sibling.poll() is None
        registered = {
            tuple(json.loads(line))
            for line in (directory / "owned-processes.jsonl").read_text().splitlines()
        }
        assert all(pid != sibling.pid for pid, _ in registered)
        for index in range(fanout):
            pid, born = json.loads((tmp_path / f"leaf-{index}.json").read_text())
            assert (pid, born) in registered
            assert (
                not psutil.pid_exists(pid)
                or psutil.Process(pid).create_time() != born
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_public_python_process_context_registers_orphaned_descendants(
    tmp_path, monkeypatch, method
):
    import psutil
    import test_author_batch_cli_learning as public

    identity = tmp_path / "leaf.json"
    program = tmp_path / "process_context.py"
    program.write_text(
        "import multiprocessing,subprocess,sys,json,psutil\nfrom pathlib import Path\n"
        "def intermediate():\n"
        "    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True)\n"
        f"    Path({str(identity)!r}).write_text(json.dumps([child.pid,"
        "psutil.Process(child.pid).create_time()]))\n"
        "    sys.exit(7)\n"
        "if __name__=='__main__':\n"
        f"    child=multiprocessing.get_context({method!r}).Process(target=intermediate)\n"
        "    child.start();child.join();assert child.exitcode==7;sys.exit(7)\n"
    )
    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        monkeypatch.setattr(public, "CLI", [sys.executable, str(program)])
        launch = subprocess.Popen

        def delayed_observation(*args, **kwargs):
            process = launch(*args, **kwargs)
            process.wait(timeout=5)
            return process

        monkeypatch.setattr(public.subprocess, "Popen", delayed_observation)
        result = _public_result(tmp_path / "context", os.environ.copy(), [], timeout=5)
        assert result.returncode == 7, result.stderr
        assert sibling.poll() is None
        pid, born = json.loads(identity.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).create_time() != born
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


def test_public_registration_failure_prevents_untracked_command(tmp_path, monkeypatch):
    import test_author_batch_cli_learning as public

    executed = tmp_path / "executed"
    monkeypatch.setattr(
        public, "OWNERSHIP_HOOK", "raise RuntimeError('registry unavailable')"
    )
    monkeypatch.setattr(
        public, "CLI", [sys.executable, "-c", f"open({str(executed)!r},'w').close()"]
    )
    result = _public_result(tmp_path / "failed-hook", os.environ.copy(), [], timeout=2)
    assert result.returncode == 125
    assert "registry unavailable" in result.stderr
    assert not executed.exists()


def test_public_shutdown_closes_generation_launch_gate(tmp_path, monkeypatch):
    import psutil
    import test_author_batch_cli_learning as public

    ready, blocked = tmp_path / "ready", tmp_path / "blocked"
    leaf = tmp_path / "leaf.py"
    leaf.write_text(
        "import signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "def stop(*args):\n"
        "    try:subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"    except RuntimeError:Path({str(blocked)!r}).touch()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(60)\n"
    )
    identity = tmp_path / "leaf.json"
    wait_ready = f"while not Path({str(ready)!r}).exists():\n time.sleep(.01)"
    script = (
        "import subprocess,sys,time,json,psutil;from pathlib import Path;"
        f"child=subprocess.Popen([sys.executable,{str(leaf)!r}],start_new_session=True);"
        f"Path({str(identity)!r}).write_text(json.dumps([child.pid,psutil.Process(child.pid).create_time()]));"
        f"exec({wait_ready!r});sys.exit(7)"
    )
    monkeypatch.setattr(public, "CLI", [sys.executable, "-c", script])
    result = _public_result(tmp_path / "shutdown", os.environ.copy(), [], timeout=5)
    assert result.returncode == 7, result.stderr
    assert blocked.exists()
    pid, born = json.loads(identity.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).create_time() != born
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_public_journal_rejects_stale_birth_for_unrelated_pid(tmp_path, monkeypatch):
    import psutil
    import test_author_batch_cli_learning as public

    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        stale = [sibling.pid, psutil.Process(sibling.pid).create_time() + 1]
        script = (
            "import os,json,sys;"
            "journal=os.environ['SMARTSOM_TEST_OWNERSHIP_JOURNAL'];"
            f"open(journal,'a').write(json.dumps({stale!r})+'\\n');sys.exit(7)"
        )
        monkeypatch.setattr(public, "CLI", [sys.executable, "-c", script])
        result = _public_result(tmp_path / "stale", os.environ.copy(), [], timeout=2)
        assert result.returncode == 7
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.parametrize("failure", ["open", "write", "flock"])
def test_public_fork_registration_failure_never_enters_user_code(
    tmp_path, monkeypatch, failure
):
    import test_author_batch_cli_learning as public

    entered, status = tmp_path / "entered", tmp_path / "status"
    program = tmp_path / "fork_failure.py"
    program.write_text(
        "import os,sys,fcntl\nfrom pathlib import Path\n"
        "parent=os.getpid()\n"
        f"module=fcntl if {failure!r}=='flock' else os\n"
        f"original=getattr(module,{failure!r})\n"
        "def fail(*args,**kwargs):\n"
        "    if os.getpid()!=parent:raise OSError('fork registration I/O failure')\n"
        "    return original(*args,**kwargs)\n"
        f"setattr(module,{failure!r},fail)\n"
        "child=os.fork()\n"
        "if child==0:\n"
        f"    Path({str(entered)!r}).touch();sys.exit(0)\n"
        "_,code=os.waitpid(child,0)\n"
        f"Path({str(status)!r}).write_text(str(os.waitstatus_to_exitcode(code)))\n"
        "sys.exit(7)\n"
    )
    monkeypatch.setattr(public, "CLI", [sys.executable, str(program)])
    result = _public_result(tmp_path / "fork-error", os.environ.copy(), [], timeout=5)
    assert result.returncode == 7, result.stderr
    assert status.read_text() == "125"
    assert not entered.exists()


@pytest.mark.parametrize("release_after_fork", [True, False])
def test_public_fork_drops_other_threads_writer_descriptor(
    tmp_path, monkeypatch, release_after_fork
):
    import psutil
    import test_author_batch_cli_learning as public

    entered, identity, status = (
        tmp_path / "entered",
        tmp_path / "identity.json",
        tmp_path / "status",
    )
    program = tmp_path / "fork_lock.py"
    program.write_text(
        "import os,sys,time,json,psutil,threading,sitecustomize\nfrom pathlib import Path\n"
        "locked=threading.Event();release=threading.Event();original=os.write\n"
        "def delayed_write(fd,payload):\n"
        "    if threading.current_thread().name=='journal-writer':\n"
        "        locked.set();assert release.wait(5)\n"
        "    return original(fd,payload)\n"
        "os.write=delayed_write\n"
        "thread=threading.Thread(target=lambda:sitecustomize.record(os.getpid()),name='journal-writer')\n"
        "thread.start();assert locked.wait(3)\n"
        "child=os.fork()\n"
        "if child==0:\n"
        f"    Path({str(entered)!r}).touch();time.sleep(60);sys.exit(0)\n"
        f"Path({str(identity)!r}).write_text(json.dumps([child,psutil.Process(child).create_time()]))\n"
        f"if {release_after_fork!r}:\n"
        "    release.set();thread.join(timeout=3);assert not thread.is_alive()\n"
        "    deadline=time.monotonic()+3\n"
        f"    while not Path({str(entered)!r}).exists():\n"
        "        assert time.monotonic()<deadline;time.sleep(.01)\n"
        "else:\n"
        "    _,code=os.waitpid(child,0)\n"
        f"    Path({str(status)!r}).write_text(str(os.waitstatus_to_exitcode(code)))\n"
        "    release.set();thread.join(timeout=3);assert not thread.is_alive()\n"
        "sys.exit(7)\n"
    )
    sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        monkeypatch.setattr(public, "CLI", [sys.executable, str(program)])
        launch = subprocess.Popen

        def delayed_observation(*args, **kwargs):
            process = launch(*args, **kwargs)
            process.wait(timeout=5)
            return process

        monkeypatch.setattr(public.subprocess, "Popen", delayed_observation)
        result = _public_result(
            tmp_path / "fork-lock", os.environ.copy(), [], timeout=5
        )
        assert result.returncode == 7, result.stderr
        assert sibling.poll() is None
        assert entered.exists() == release_after_fork
        if not release_after_fork:
            assert status.read_text() == "125"
        pid, born = json.loads(identity.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).create_time() != born
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


def test_public_creator_registration_failure_closes_native_child(tmp_path, monkeypatch):
    import psutil
    import test_author_batch_cli_learning as public

    identity = tmp_path / "native.json"
    script = (
        "import os,sys,json,psutil,sitecustomize,subprocess;from pathlib import Path;"
        'exec("def fail(pid):\\n'
        f" Path({str(identity)!r}).write_text(json.dumps([pid,psutil.Process(pid).create_time()]))\\n"
        " raise OSError('creator registration failed')\");"
        "sitecustomize.record=fail;"
        "exec(\"try:subprocess.Popen(['/bin/sleep','60'])\\n"
        'except OSError:sys.exit(7)")'
    )
    monkeypatch.setattr(public, "CLI", [sys.executable, "-c", script])
    result = _public_result(
        tmp_path / "native-failure", os.environ.copy(), [], timeout=5
    )
    assert result.returncode == 7, result.stderr
    pid, born = json.loads(identity.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).create_time() != born
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_public_ancestry_rejects_reused_parent_identity(monkeypatch):
    from types import SimpleNamespace

    import psutil
    import test_author_batch_cli_learning as public

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return {101: 2.0, 202: 3.0}[self.pid]

        def ppid(self):
            return {101: 0, 202: 101}[self.pid]

        def is_running(self):
            return True

        def children(self, recursive=True):
            return [Process(202)] if self.pid == 101 else []

    # The PID is live, but belongs to a newer parent than the recorded identity.
    # Its genuine child must not become eligible for this command's cleanup.
    owned = public._OwnedProcesses({(101, 1.0): Process(101)}, None)
    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda attrs: iter(
            [
                SimpleNamespace(pid=101, info={"ppid": 0}),
                SimpleNamespace(pid=202, info={"ppid": 101}),
            ]
        ),
    )
    public._remember_children(owned)
    assert set(owned) == {(101, 1.0)}


def inputs(directory, algorithm):
    path = _inputs(directory, "rllib")
    factory = yaml.safe_load(
        (ROOT / "src/smartsom/studio/templates/template_010.yaml").read_text(
            encoding="utf-8"
        )
    )
    for buffer in factory["factory"]["buffers"]:
        if buffer["role"] == "system_input":
            buffer["storage"]["capacity"] = 10
    _write(directory / "factory.yaml", factory)
    method = yaml.safe_load((directory / "algorithm.yaml").read_text(encoding="utf-8"))
    method["learner"].update(
        algorithm=algorithm,
        parameters={
            "batch_size": 2,
            **(
                {"n_epochs": 1}
                if algorithm == "ppo"
                else {"warmup_ticks": 0, "replay_capacity": 16, "train_every_ticks": 1}
            ),
        },
    )
    for role in ("machine", "buffer", "dispatcher"):
        extensions = method["agents"][role]["default"]["extensions"]
        extensions.update(
            observation={
                "name": "builtin.physical_job",
                "version": "1",
                "parameters": {"include_inspection": True},
            },
            network_implementation={
                "name": "builtin.physical_job_candidate",
                "version": "1",
            },
            network={
                branch: {"hidden_sizes": [8]}
                for branch in (("actor", "critic") if algorithm == "ppo" else ("q",))
            },
        )
    method["agents"]["mover"]["default"]["name"] = "automatic_travel"
    _write(directory / "algorithm.yaml", method)
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["training"].update(total_ticks=8, ticks_per_update=4, max_ticks=4)
    experiment["runtime"]["environment"].update(tick_limit=4)
    _write(
        directory / "travel.yaml",
        {"schema": "smartsom.travel-time-matrix/v1", "source": "auto", "overrides": {}},
    )
    experiment["runtime"]["environment"]["transport"] = {
        "mode": "travel_time_matrix",
        "matrix": "../travel.yaml",
    }
    experiment["validation"].update(
        every_updates=None, updates=[1, 2], best_mode="all_complete"
    )
    experiment["evaluation"].update(
        checkpoint="best",
        no_eligible_best={
            "outcome": "record_unselected",
            "last_diagnostic": {"enabled": True},
        },
    )
    experiment["checkpointing"] = {"retention": {"mode": "latest_full_and_best"}}
    experiment["execution"] = {
        "calibration_level": "off",
        "scheduling": "fixed",
        "max_concurrent": 2,
        "background": False,
    }
    _write(path, experiment)
    # Only Experiment YAMLs belong in the scanned directory.
    experiments = directory / "experiments"
    experiments.mkdir()
    experiment.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    _write(experiments / "experiment.yaml", experiment)
    return experiments


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_public_batch_run_without_calibration_and_unselected_final(tmp_path, algorithm):
    _require_cpu()
    directory = inputs(tmp_path / "inputs", algorithm)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = _public_result(
        tmp_path / "public-run",
        env,
        [
            "batch-run",
            str(directory),
            "--no-background",
            "--calibration-level",
            "off",
            "--log-format",
            "json",
            "--progress",
            "off",
        ],
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = tmp_path / "outputs"
    selections = list(outputs.rglob("selection-outcome.json"))
    assert selections
    outcome = json.loads(selections[0].read_text(encoding="utf-8"))
    assert all(
        json.loads(path.read_text(encoding="utf-8")) == outcome for path in selections
    )
    assert outcome["status"] == "no_eligible_best"
    assert outcome["final"] is None and not outcome["ranking_eligible"]
    assert not list(outputs.rglob("tuning-final.json"))
    diagnostic = json.loads(
        next(outputs.rglob("tuning-final-last-diagnostic.json")).read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic and all(
        row["evaluation_label"] == "FINAL-LAST-DIAGNOSTIC"
        and not row["ranking_eligible"]
        for row in diagnostic
    )
    calibrations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs.rglob("calibration.json")
    ]
    assert calibrations and all(
        not record["measurements"] and record["calibration_level"] == "off"
        for record in calibrations
    )
    for path in outputs.rglob("retention.json"):
        if (
            path.parent.name == "checkpoints"
            and (path.parent / "retention-owner.json").exists()
            and path.parent.parent.name != "support"
        ):
            from smartsom.experiments.checkpoint_retention import manifest

            value = manifest(path.parent.parent)
            assert value["updates"] == 2
            assert value["selected_best"] is None
            assert len(list(path.parent.glob("update-*"))) == 1


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_declared_policy_spawn_worker_count_preserves_logical_streams(
    tmp_path, algorithm
):
    _require_cpu()
    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.composable import TrainingSession, allocate

    path = inputs(tmp_path / "inputs", algorithm) / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["runtime"]["num_envs"] = 2
    experiment["validation"]["enabled"] = False
    experiment["validation"].pop("updates")
    experiment["validation"]["every_updates"] = 1
    experiment["evaluation"].pop("no_eligible_best")
    experiment["evaluation"]["checkpoint"] = "last"
    observed = []
    for processes in (1, 2):
        experiment["runtime"]["sampling_processes"] = processes
        experiment["output"]["name"] = f"declared-{algorithm}-{processes}"
        _write(path, experiment)
        prepared = (
            compile_experiment(path, require_dependencies=True).entries[0].prepared
        )
        root, record, prepared = allocate(prepared, "training")
        session = TrainingSession(prepared, root, record)
        try:
            while not session.training_done:
                session.step_update()
            observed.append(
                (
                    session.actions,
                    {
                        key: value.fingerprint()
                        for key, value in session.policies.items()
                        if hasattr(value, "fingerprint")
                    },
                    session.optimizations,
                )
            )
        finally:
            session.close()
    assert observed[0] == observed[1]


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_public_committed_stop_and_fresh_process_resume(tmp_path, algorithm):
    _require_cpu()
    from smartsom.experiments.tuning_session import verify_commit

    directory = inputs(tmp_path / "inputs", algorithm)
    path = directory / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["training"].update(total_ticks=512, ticks_per_update=64, max_ticks=512)
    experiment["runtime"]["environment"]["tick_limit"] = 512
    experiment["validation"]["updates"] = [2, 5, 8]
    experiment["execution"]["max_concurrent"] = 1
    _write(path, experiment)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUTF8="1",
    )
    command = [sys.executable, "-m", "smartsom.experiments.cli"]
    display = ["--log-format", "json", "--progress", "off"]
    child = None
    with _command(
        tmp_path / "public-stop",
        env,
        *(
            [
                "batch-run",
                str(directory),
                "--no-background",
                "--calibration-level",
                "off",
                *display,
            ]
        ),
        register_exit_children=True,
    ) as execution:
        child, owned, stdout, stderr = execution

        def log_text():
            return stdout.read_text() + stderr.read_text()

        try:
            deadline = time.monotonic() + 180
            root = None
            while time.monotonic() < deadline and child.poll() is None:
                _remember_children(owned)
                for manifest in (tmp_path / "outputs").rglob("batch.json"):
                    state = json.loads(manifest.read_text(encoding="utf-8"))
                    if any(
                        row.get("checkpoint") and 0 < row.get("updates", 0) < 8
                        for row in state.get("entries", {}).values()
                    ):
                        root = manifest.parent
                        break
                if root is not None:
                    break
                time.sleep(0.1)
            assert root is not None, log_text()
            owner = json.loads(
                (root / "control/owner.json").read_text(encoding="utf-8")
            )
            driver_root = Path(owner["driver_root"])
            stop = subprocess.run(
                [*command, "stop", str(driver_root), "--timeout", "60"],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
            assert stop.returncode == 0, stop.stdout + stop.stderr
            child.wait(timeout=90)
            state = json.loads((root / "batch.json").read_text(encoding="utf-8"))
            assert state["status"] == "interrupted", log_text()
            assert (
                json.loads((driver_root / "batch.json").read_text(encoding="utf-8"))[
                    "status"
                ]
                == "stopped"
            )
            before = next(iter(state["entries"].values()))
            boundary = verify_commit(before["checkpoint"])
            assert 0 < boundary["updates"] < 8
            assert boundary["physical_ticks"] == boundary["updates"] * 64
        finally:
            if child.poll() is None:
                # Cooperative cancellation belongs only to this disposable test run.
                if root is not None:
                    subprocess.run(
                        [
                            *command,
                            "stop",
                            str(
                                json.loads(
                                    (root / "control/owner.json").read_text(
                                        encoding="utf-8"
                                    )
                                )["driver_root"]
                            ),
                            "--timeout",
                            "30",
                        ],
                        env=env,
                        capture_output=True,
                        timeout=60,
                    )
                # The context's exact-identity emergency cleanup runs even when
                # admission never reached a checkpoint. Do not mask the original
                # boundary failure with an additional unowned wait timeout.
    resumed = _public_result(
        tmp_path / "public-resume",
        env,
        ["tune", "resume", str(root), *display],
        # Linux's last evaluation slope predicts ~320 s plus publication/shutdown.
        # 480 s leaves margin over ~340 s; 120 s without semantic counters is four
        # display periods and more than twice the observed validation duration.
        timeout=480,
        stalled=120,
        progress_path=root / "logs/progress.json",
        entry_id=next(iter(state["entries"])),
        restored={
            "physical_ticks": boundary["physical_ticks"],
            "updates": boundary["updates"],
        },
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed" and all(
        row["status"] == "completed" for row in state["entries"].values()
    )
    entry = next(iter(state["entries"].values()))
    checkpoint = Path(entry["checkpoint"])
    marker = verify_commit(checkpoint)
    assert (
        marker["phase"] == "experiment_complete"
        and marker["updates"] == 8
        and marker["physical_ticks"] == 512
    )
    continuation = pickle.loads((checkpoint / "continuation.pkl").read_bytes())
    assert continuation["validation_phase"] == {
        "pending_update": None,
        "completed_updates": [2, 5, 8],
    }
    assert all(
        continuation["optimizations"][role] > 0
        for role in ("machine", "buffer", "dispatcher")
    )
    assert len(continuation["history"]) == 8 and len(continuation["actions"]) == 512
    original = json.loads(
        (checkpoint / "original-prepared.json").read_text(encoding="utf-8")
    )
    for declaration in json.loads(original["policies_json"]).values():
        if declaration["role"] != "mover":
            extensions = declaration["implementation"]["extensions"]
            assert extensions["observation"]["name"] == "builtin.physical_job"
            assert (
                extensions["network_implementation"]["name"]
                == "builtin.physical_job_candidate"
            )
            assert (
                extensions["observation"]["code_sha256"]
                == extensions["network_implementation"]["code_sha256"]
            )
    calibrations = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in root.rglob("calibration.json")
    ]
    assert calibrations and all(
        not record["measurements"] and record["calibration_level"] == "off"
        for record in calibrations
    )


def _snapshot(**values):
    return {
        "schema": "smartsom.runtime-progress/v1",
        "tasks": [{"id": "entry", "values": values}],
    }


def test_resume_watchdog_ignores_heartbeats_replays_and_wrong_entries():
    timer = _ResumeDeadline(
        "entry",
        started=10,
        total=480,
        stalled=120,
        restored={"physical_ticks": 128, "updates": 2},
    )
    timer.observe(_snapshot(physical_ticks=128), 10)
    for elapsed in (30, 60, 119):
        snapshot = _snapshot(
            physical_ticks=128,
            updated_at=elapsed,
            available_cpus=elapsed,
            validation_round=1,
        )
        snapshot["updated_at"] = elapsed
        snapshot["tasks"][0]["stage"] = f"heartbeat-{elapsed}"
        assert timer.observe(snapshot, 10 + elapsed)
    assert timer.last_progress == 10
    wrong = _snapshot(physical_ticks=512)
    wrong["tasks"][0]["id"] = "other"
    assert timer.observe(wrong, 110) and timer.last_progress == 10
    assert not timer.observe(wrong, 130)
    assert timer.reason == "no semantic progress deadline"


def test_resume_watchdog_tracks_real_validation_round_ticks_and_rejects_stale_ticks():
    timer = _ResumeDeadline("entry", started=0, total=480, stalled=120)
    assert timer.observe(
        _snapshot(validation_round=1, validation_tick=512, validation_case_active=True),
        30,
    )
    assert timer.observe(
        _snapshot(
            validation_round=2, validation_tick=512, validation_case_active=False
        ),
        60,
    )
    assert timer.last_progress == 30
    assert timer.observe(
        _snapshot(validation_round=2, validation_tick=1, validation_case_active=True),
        90,
    )
    assert timer.last_progress == 90
    assert timer.observe(
        _snapshot(validation_round=1, validation_tick=512, validation_case_active=True),
        150,
    )
    assert timer.last_progress == 90
    assert timer.observe(_snapshot(validation_batches_finished=2), 180)
    assert timer.last_progress == 180
    assert timer.observe(_snapshot(evaluation_case_active=True, evaluation_tick=1), 200)
    assert timer.observe(_snapshot(evaluation_case_active=True, evaluation_tick=1), 230)
    assert timer.last_progress == 200
    assert timer.observe(_snapshot(evaluation_finished=1), 260)
    assert timer.last_progress == 260


def test_resume_watchdog_hard_cap_wins_despite_continuous_real_progress():
    timer = _ResumeDeadline("entry", started=0, total=480, stalled=120)
    for index, now in enumerate(range(30, 480, 30), start=1):
        assert timer.observe(_snapshot(physical_ticks=index), now)
    assert not timer.observe(_snapshot(physical_ticks=512), 480)
    assert timer.reason == "hard total deadline" and timer.last_progress == 450
    assert not timer.observe({}, 481)


def test_resume_watchdog_bounds_counters_and_treats_missing_evidence_as_stall(tmp_path):
    timer = _ResumeDeadline("entry", started=0, total=480, stalled=120)
    assert timer.observe(
        _snapshot(
            physical_ticks=513,
            updates=True,
            validation_batches_finished=-1,
            validation_round=100,
            validation_tick=512,
            validation_case_active=True,
        ),
        119,
    )
    assert timer.maxima == {} and timer.last_progress == 0
    missing = tmp_path / "progress.json"
    assert _progress_snapshot(missing) == {}
    missing.write_text('{"tasks": [null]}')
    assert _progress_snapshot(missing) == {}
    missing.write_bytes(b"x" * 262145)
    assert _progress_snapshot(missing) == {}
    assert not timer.observe(_progress_snapshot(missing), 120)


@pytest.mark.parametrize("journalled", [True, False])
def test_wait_scan_cadence_and_fresh_identity_check_at_boundary(
    monkeypatch, tmp_path, journalled
):
    from types import SimpleNamespace

    import test_author_batch_cli_learning as public

    now, scans = [0.0], []
    owned = SimpleNamespace(
        registry=object() if journalled else None, registry_pending=False
    )
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(public.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        public.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(
        public, "_remember_children", lambda value: scans.append(now[0])
    )
    public._wait(
        (process, owned, tmp_path / "stdout", tmp_path / "stderr"),
        lambda: now[0] >= 1.01,
        timeout=5,
    )
    assert scans[-1] == now[0]  # boundary falls between the periodic scans
    assert len(scans) <= 6 if journalled else len(scans) >= 50


@pytest.mark.parametrize("hard", [False, True])
def test_resume_watchdog_failure_preserves_context_cleanup_and_summary(
    monkeypatch, tmp_path, hard
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import test_author_batch_cli_learning as public

    now, released = [0.0], []
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps(_snapshot(physical_ticks=1)))
    process = SimpleNamespace(poll=lambda: None)
    owned = SimpleNamespace(registry=object(), registry_pending=False)

    @contextmanager
    def command(directory, environment, *arguments, **kwargs):
        directory.mkdir()
        stdout, stderr = directory / "stdout.log", directory / "stderr.log"
        stdout.write_text("original output")
        stderr.write_text("original diagnostic")
        try:
            yield process, owned, stdout, stderr
        finally:
            released.append(True)

    def sleep(seconds):
        now[0] += seconds
        values = (
            {"physical_ticks": int(now[0] * 100) + 1}
            if hard
            else {"updated_at": now[0]}
        )
        progress.write_text(json.dumps(_snapshot(**values)))

    monkeypatch.setattr(public.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(public.time, "sleep", sleep)
    monkeypatch.setattr(public, "_remember_children", lambda value: None)
    monkeypatch.setattr(sys.modules[__name__], "_command", command)
    reason = "hard total" if hard else "no semantic progress"
    with pytest.raises(pytest.fail.Exception, match=reason) as error:
        _public_result(
            tmp_path / "command",
            {},
            [],
            timeout=1,
            stalled=2 if hard else 0.5,
            progress_path=progress,
            entry_id="entry",
        )
    assert "original diagnostic" in str(error.value) and released == [True]
    summary = json.loads((tmp_path / "command/deadline-summary.json").read_text())
    assert summary["reason"].startswith(reason)
    assert summary["elapsed_seconds"] <= 1.03


def test_wait_rechecks_boundary_after_fresh_journal_scan(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import test_author_batch_cli_learning as public

    now, scans, boundary_checks = [0.0], [], []
    owned = SimpleNamespace(registry=object(), registry_pending=False, remaining=False)

    def remember(value):
        scans.append(now[0])
        # A child identity arrives between periodic scans and the requested boundary.
        if 0.1 <= now[0] < 0.2:
            value.remaining = True
        elif now[0] >= 0.2:
            value.remaining = False

    def boundary():
        boundary_checks.append((now[0], owned.remaining))
        return now[0] >= 0.1 and not owned.remaining

    monkeypatch.setattr(public.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        public.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(public, "_remember_children", remember)
    public._wait(
        (
            SimpleNamespace(poll=lambda: None),
            owned,
            tmp_path / "stdout",
            tmp_path / "stderr",
        ),
        boundary,
        timeout=5,
    )
    assert now[0] >= 0.2
    assert any(
        0.1 <= moment < 0.2 and remaining for moment, remaining in boundary_checks
    )
    assert scans[-1] == now[0]
