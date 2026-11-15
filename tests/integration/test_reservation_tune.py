"""Official Tune cohorts release phase resources and restore committed work."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_public_rule_random_fixed_panel_and_completed_restore(tmp_path):
    import json

    import yaml

    from smartsom.algorithms.base_policy import build_policy, evaluation_recipe
    from smartsom.config.base_diagnostic import generate_scenario, load_inputs
    from smartsom.config.codec import digest, primitive
    from smartsom.experiments.reservation_runtime import episode
    from tests.helpers.rllib_runtime import author_runtime, command

    path, profile = author_runtime(tmp_path / "inputs", evaluation=False)
    document = yaml.safe_load(path.read_text())
    document["run"].update(
        task="evaluate", methods=["rule", "random"], split="test", seconds=None
    )
    document["concurrency"] = {"evaluation_processes": 2}
    path.write_text(yaml.safe_dump(document))
    ex, factory, workload, options, _ = load_inputs(path)
    cases = [
        {
            "id": f"world-{index}",
            "condition": condition,
            "seed": 9100 + index,
            "action_seeds": [31, 47],
            "scenario": primitive(
                generate_scenario(
                    *ex.inputs_for(condition, factory, workload),
                    9100 + index,
                    condition,
                    ex.drain_ticks,
                )
            ),
        }
        for index, condition in enumerate(("normal", "low", "medium", "high"))
    ]
    panel = path.parent / "test.json"
    panel.write_text(
        json.dumps(
            {
                "schema": "smartsom.fixed-cases/v1",
                "split": "test",
                "factory_sha256": digest(primitive(factory)),
                "workload_sha256": digest(workload.model_dump(mode="json")),
                "cases": cases,
            }
        )
    )
    document["case_sets"] = {"test": "test.json"}
    path.write_text(yaml.safe_dump(document))
    profile.write_text(
        yaml.safe_dump(
            {
                "resources": {"cpus": 3},
                "rllib": {"evaluation_num_env_runners": 2},
            }
        )
    )
    command("check", path, "--profile", profile)
    result = command("run", path, "--profile", profile, timeout=120)
    (tmp_path / "public.log").write_text(result.stdout + result.stderr)
    root = path.parent / "run"
    rows = [
        json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 16
    assert len(list((root / "case-commits").glob("case-*.json"))) == 16
    assert not list(root.glob("checkpoint-refs-*.json"))
    assert not (root / "last.pt").exists()
    assert {r["evaluation_identity"]["policy_seed"] for r in rows} == {31, 47}
    from smartsom.config.scenario_snapshot import scenario_from_snapshot

    by_world = {digest(case["scenario"]): case for case in cases}
    for row in rows:
        identity = row["evaluation_identity"]
        case = by_world[identity["world_sha256"]]
        assert not identity["learning"] and identity["checkpoint_sha256"] is None
        recipe = evaluation_recipe(options, row["method"])
        direct, _, _ = episode(
            *ex.inputs_for(case["condition"], factory, workload),
            recipe,
            build_policy(recipe, identity["policy_seed"], training=False),
            case["seed"],
            case["condition"],
            ex.drain_ticks,
            scenario=scenario_from_snapshot(case["scenario"]),
            metrics_level="off",
            task_objective=True,
        )
        for key in (
            "tick",
            "completed",
            "failed",
            "makespan",
            "mean_tardiness",
            "joint_cost",
            "termination_reason",
        ):
            assert row[key] == direct[key]
        assert row["objective_schema"] == "smartsom.normalized-task/v1"
        if row["joint_cost"] is not None:
            assert row["joint_cost"] == pytest.approx(
                row["makespan"] / 5000
                + row["mean_tardiness"] / 300
                + (1000 / 300) * (1 - row["final_passing_rate"])
            )
    checkpoint = next(root.rglob("scientific-task.json"))
    payload = json.loads(checkpoint.read_text())
    assert payload["state"]["phase"] == "complete"
    assert {"cases.jsonl", "summary.json", "research-summary.json"} <= payload[
        "files"
    ].keys()
    files = [
        root / "cases.jsonl",
        root / "summary.json",
        *root.glob("tracking/*.jsonl"),
    ]
    before = {p: p.read_bytes() for p in files}
    command("resume", root, timeout=90)
    assert all(p.read_bytes() == content for p, content in before.items())


SCRIPT = r"""
import json, os, sys
from pathlib import Path
from ray.tune import Callback
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation import allocate
from smartsom.experiments.tune_execution import fit_cohort
from tests.helpers.rllib_runtime import author_runtime, children, assert_released
root, repository=Path(sys.argv[1]),Path(sys.argv[2])
roots=[]
for seed in (101,102,103):
    path,profile=author_runtime(root/f'input-{seed}',ticks=80,evaluation=True)
    plan=compile_reservation(path,profile=profile,seed=seed)
    roots.append(allocate(plan,root=root/f'seed-{seed}',announce=False))
class Observe(Callback):
    def __init__(self):self.active=set();self.events=[];self.peak=0;self.processes=set()
    def on_trial_start(self,iteration,trials,trial,**info):
        self.active.add(trial.trial_id);self.peak=max(self.peak,len(self.active));self.events.append(('start',trial.trial_id))
    def on_checkpoint(self,iteration,trials,trial,checkpoint,**info):
        self.processes.update(children(os.getpid()))
    def on_trial_complete(self,iteration,trials,trial,**info):
        self.active.remove(trial.trial_id);self.events.append(('complete',trial.trial_id))
observer=Observe()
results=fit_cohort(roots,storage=root/'tune/cohort',resources={'cpus':4},max_concurrent=2,callbacks=[observer])
assert not results.errors and len(results)==3
assert observer.peak==2 and not observer.active
assert [event for event,_ in observer.events].index('complete')< [i for i,(event,_) in enumerate(observer.events) if event=='start'][2]
for result in results:
    assert result.metrics['done'] and result.metrics['algorithm_iteration']>0
    assert len(result.metrics['final_evaluation']['last']['cases'])==4
    assert result.metrics['final_evaluation']['last']['selection_mode']=='sample'
assert observer.processes
assert_released(observer.processes)
before={p:p.read_bytes() for directory in roots for p in [directory/'progress.json',*directory.glob('checkpoint-refs-*.json'),*directory.glob('tracking/*.jsonl')]}
restored=fit_cohort(roots,storage=root/'tune/cohort',resources={'cpus':4},max_concurrent=2,resume=True)
assert len(restored)==3 and not restored.errors
assert all(p.read_bytes()==content for p,content in before.items())
assert not sys.modules['ray'].is_initialized()
"""


def test_official_cohort_and_whole_run_restore(tmp_path):
    pytest.importorskip("ray")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), str(ROOT)],
        cwd=ROOT,
        env=dict(
            os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_USAGE_STATS_ENABLED="0"
        ),
        text=True,
        capture_output=True,
        timeout=240,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_json_run_keeps_backend_output_out_of_events(tmp_path):
    import json

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from smartsom.experiments.cli import main; raise SystemExit(main(sys.argv[1:]))",
            "run",
            "experiments/examples/current-smoke/default/experiment.yaml",
            "--output-root",
            str(tmp_path),
            "--cpus",
            "3",
            "--verbose",
            "1",
            "--no-debug",
            "--log-format",
            "json",
        ],
        cwd=ROOT,
        env=dict(
            os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_USAGE_STATS_ENABLED="0"
        ),
        text=True,
        capture_output=True,
        timeout=90,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert all("events" not in event for event in events)
    assert not any(event.get("type") == "warning" for event in events)
    assert '"status": "completed"' in result.stdout
    assert "\x1b" not in result.stderr


RETRY_SCRIPT = r"""
import json, sys
from pathlib import Path
import ray.cloudpickle as pickle
from tests.helpers.rllib_runtime import author_runtime
from smartsom.experiments import rllib_trainable
from smartsom.experiments.cli import main
from tests.integration.test_rllib_dqn_checkpoint import assert_state_equal
root, repository = Path(sys.argv[1]), Path(sys.argv[2])
limit, exhaust, method = int(sys.argv[3]), sys.argv[4]=='exhaust', sys.argv[5]
action=sys.argv[6]
path, profile = (author_runtime(root/'inputs', method, ticks=1000 if method=='dqn' else 200, evaluation=False)
                 if action=='run' else (root/'inputs/experiment.yaml', root/'inputs/profile.yaml'))
output=path.parent/'run'
marker=root/'injected-failure'
class FailOnce(rllib_trainable.RLlibTrainable):
    def step(self):
        if self.algorithm is not None and self.algorithm.iteration==1 and (exhaust or not marker.exists()):
            if method=='dqn':
                state=self.algorithm.get_state()
                with (root/'expected.pkl').open('wb') as stream:
                    pickle.dump({key:state[key] for key in ('learner_group','replay_buffer')},stream)
                assert self.algorithm.local_replay_buffer.get_num_timesteps()>0
            with (root/'failures.jsonl').open('a') as stream:stream.write(json.dumps({'attempt':self.execution_id})+'\n')
            marker.write_text(self.trial_id)
            raise RuntimeError('injected after official committed iteration')
        return super().step()
    def load_checkpoint(self, checkpoint_dir):
        super().load_checkpoint(checkpoint_dir)
        if method=='dqn':
            with (root/'expected.pkl').open('rb') as stream:expected=pickle.load(stream)
            actual=self.algorithm.get_state()
            for key in expected:assert_state_equal(expected[key],actual[key])
            assert self.algorithm.iteration==1
            (root/'verified-restore').write_text(self.trial_id)
rllib_trainable.RLlibTrainable=FailOnce
code=main(['run',str(path),'--profile',str(profile),'--progress','off','--set',f'experiment.run.max_failures={limit}'] if action=='run' else
          ['retry',str(output),'--all','--foreground','--progress','off'])
assert marker.exists()
if exhaust:
    assert code!=0
    assert len((root/'failures.jsonl').read_text().splitlines())==2
elif limit==0 and action=='run':
    assert code!=0
    assert len((root/'failures.jsonl').read_text().splitlines())==1
else:
    assert code==0
    report=json.loads((output/'batch.json').read_text())
    assert report['status']=='completed' and report['algorithm_iteration']>1
    if method=='dqn':assert (root/'verified-restore').read_text()==marker.read_text()
assert not sys.modules['ray'].is_initialized()
"""


@pytest.mark.parametrize(
    "max_failures,mode,method",
    [(1, "once", "ppo"), (1, "exhaust", "ppo"), (0, "once", "dqn")],
)
def test_real_fixed_trial_default_failure_and_official_retry(
    tmp_path, max_failures, mode, method
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            RETRY_SCRIPT,
            str(tmp_path),
            str(ROOT),
            str(max_failures),
            mode,
            method,
            "run",
        ],
        cwd=ROOT,
        env=dict(
            os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_USAGE_STATS_ENABLED="0"
        ),
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr

    if max_failures == 0:
        from tests.helpers.rllib_runtime import environment

        output = tmp_path / "inputs/run"
        progress = (output / "progress.json").read_bytes()
        resumed = subprocess.run(
            [
                sys.executable,
                "-m",
                "smartsom.experiments.cli",
                "resume",
                str(output),
                "--foreground",
                "--progress",
                "off",
            ],
            env=environment(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resumed.returncode != 0
        assert (output / "progress.json").read_bytes() == progress
        retried = subprocess.run(
            [
                sys.executable,
                "-c",
                RETRY_SCRIPT,
                str(tmp_path),
                str(ROOT),
                str(max_failures),
                mode,
                method,
                "retry",
            ],
            cwd=ROOT,
            env=environment(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        (tmp_path / "retry-driver.log").write_text(retried.stdout + retried.stderr)
        assert retried.returncode == 0, retried.stdout + retried.stderr
