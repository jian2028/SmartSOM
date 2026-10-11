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


def _public_result(directory, environment, arguments, *, timeout):
    """Keep exact descendant identities until timeout cleanup finishes.

    subprocess.run kills only the CLI on timeout, leaving its native/Ray child
    driver running. Preserve the failure and release only this command's owned
    process tree before another case measures available resources.
    """
    with _command(
        directory, environment, *arguments, register_exit_children=True
    ) as command:
        process, owned, stdout, stderr = command
        _wait(
            command,
            lambda: process.poll() is not None,
            timeout=timeout,
            allow_exit=True,
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
        timeout=300,
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
