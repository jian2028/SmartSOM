"""Genuine optional Ray execution; no manufactured performance measurements."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("resource_execution", ["adaptive", "fixed"])
def test_real_base_tune_two_seeds(tmp_path, resource_execution):
    pytest.importorskip("ray")
    script = r"""
import json, sys
from pathlib import Path
from smartsom import api
root = Path(sys.argv[1])
config = Path(sys.argv[2])
shared = ["workload.orders=8", "workload.initial_orders=4", "workload.mean_interarrival=4",
          "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}", "algorithm.epochs=1",
          "experiment.run.seconds=120"]
api.run_experiment(config, task="calibrate", sets=shared + [f'experiment.run.output="{root}/calibration"'])
result = api.run_experiment(config, task="train", tuning="auto", sets=shared + [
    f'experiment.run.output="{root}/train"',
    f'experiment.run.calibration="{root}/calibration/calibration.json"',
    "experiment.train_seeds=[101,102]", "experiment.train_ticks=80",
    "experiment.run.probe_ticks=80", "experiment.run.calibration_seconds=90",
    "experiment.run.max_concurrent=2",
    f"experiment.run.resource_execution={sys.argv[3]}"])
assert result["status"] == "completed", result
assert sorted(r["seed"] for r in result["seeds"]) == [101, 102]
assert all(r["physical_ticks"] == 80 and not r["error"] for r in result["seeds"])
from smartsom.experiments.reservation import execute_saved
assert execute_saved(root / "train") == {"directory": str(root / "train"), **result}
assert api.resume(root / "train") == {"directory": str(root / "train"), **result}
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(ROOT / "configs/base-diagnostic/small.yaml"),
            resource_execution,
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads((tmp_path / "train/calibration-performance.json").read_text())
    assert any(row["valid"] and row["throughput"] > 0 for row in proof["measurements"])


def test_real_base_resize_restores_committed_rollouts(tmp_path):
    pytest.importorskip("ray")
    script = r"""
import json, sys
from pathlib import Path
import ray
original_init=ray.init
def two_cpu_test_runtime(**kwargs):
    kwargs['num_cpus']=max(2,kwargs.get('num_cpus',0))
    return original_init(**kwargs)
ray.init=two_cpu_test_runtime
from smartsom import api
from smartsom.experiments import tuning_ray
root=Path(sys.argv[1]); config=Path(sys.argv[2])
original=tuning_ray.build_tuner
changes={}
def build(entries, **kwargs):
    broker=kwargs['broker']
    # Fault-injection acceptance explicitly requests actor replacement; normal
    # adaptive scheduling now keeps the measured thread layout fixed.
    for identity, profiles in broker.profiles.items():
        template=profiles[0]
        broker.profiles[identity]=[{**template, 'CPU': t} for t in (1,2)]
    def allocation(controller, trial, result, scheduler):
        identity=trial.config['experiment_id']
        broker.note_actor(identity,trial)
        broker.refresh()
        count=changes.get(identity,0)
        if count>=2:
            return None
        current=int(trial.placement_group_factory.required_resources['CPU'])
        target=next(p for p in broker.profiles[identity] if p['CPU'] != current)
        changes[identity]=count+1
        return {k:target[k] for k in ('CPU','GPU','memory')}
    kwargs['allocation_function']=allocation
    return original(entries, **kwargs)
tuning_ray.build_tuner=build
shared=['workload.orders=8','workload.initial_orders=4','workload.mean_interarrival=4',
'workload.base_ticks={A: 2, B: 2, C: 2, D: 2}','algorithm.epochs=1','experiment.run.seconds=180']
api.run_experiment(config,task='calibrate',sets=shared+[f'experiment.run.output="{root}/calibration"'])
result=api.run_experiment(config,task='train',tuning='auto',sets=shared+[
f'experiment.run.output="{root}/train"',f'experiment.run.calibration="{root}/calibration/calibration.json"',
'experiment.train_seeds=[101]','experiment.train_ticks=1200','experiment.run.probe_ticks=80',
'experiment.run.calibration_seconds=90','experiment.run.max_concurrent=1'])
assert result['status']=='completed',result
acks=[json.loads(p.read_text()) for p in (root/'train/seeds/101/allocations').glob('*.json')]
acks.sort(key=lambda a:a['allocation_epoch'])
assert len(acks)>=3,acks
assert {a['threads'] for a in acks}=={1,2},acks
assert len({a['pid'] for a in acks})>=3,acks
assert acks[1]['physical_ticks']>0 and acks[2]['physical_ticks']>acks[1]['physical_ticks'],acks
assert result['seeds'][0]['physical_ticks']==1200,result
rows=[json.loads(s) for s in (root/'train/seeds/101/episodes.jsonl').read_text().splitlines()]
assert [r['episode'] for r in rows]==list(range(1,len(rows)+1))
assert len({r['physical_ticks'] for r in rows})==len(rows)
(root/'resize-proof.json').write_text(json.dumps({'acks':acks,'result':result},indent=2))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(ROOT / "configs/base-diagnostic/small.yaml"),
        ],
        text=True,
        capture_output=True,
        timeout=300,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_completed_probe_receipts_survive_supervisor_restart(tmp_path, monkeypatch):
    from smartsom.experiments.reservation_tune import ResumableProbeSupervisor
    from smartsom.experiments.tuning_calibration import CandidateMeasurement
    from smartsom.experiments.tuning_probe import ProbeSupervisor
    from smartsom.experiments.tuning_resources import ExecutionProfile

    calls = []
    profile = ExecutionProfile(1, 1)

    def measured(self, probe, group, selected, remaining, cancelled):
        calls.append(1)
        return CandidateMeasurement(selected, 10.0, 100, elapsed_seconds=2)

    monkeypatch.setattr(ProbeSupervisor, "run", measured)
    for _ in range(2):
        supervisor = ResumableProbeSupervisor(work_root=tmp_path)
        assert supervisor.run(None, {"id": "same"}, profile, 10).throughput == 10
    assert len(calls) == 1
    receipt = next((tmp_path / "receipts").glob("*.json"))
    data = json.loads(receipt.read_text())
    data["measurement"]["throughput"] = 1000
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupted"):
        ResumableProbeSupervisor(work_root=tmp_path).run(
            None, {"id": "same"}, profile, 10
        )


def test_real_tune_pause_releases_actor_and_resumes_native_state(tmp_path):
    pytest.importorskip("ray")
    script = r"""
import json,sys
from pathlib import Path
from ray.tune import Callback
from smartsom import api
from smartsom.experiments import tuning_ray, runtime_control
root=Path(sys.argv[1]); config=Path(sys.argv[2]); events=[]
original=tuning_ray.build_tuner
class PauseOnce(Callback):
    requested=False
    resumed=False
    def on_trial_result(self, iteration, trials, trial, result, **info):
        if not self.requested and not result.get('done') and result.get('physical_ticks',0)>0:
            runtime_control.submit(root/'train','pause',task_id='101')
            self.requested=True
            events.append({'event':'requested','ticks':result['physical_ticks']})
    def on_step_begin(self, iteration, trials, **info):
        if self.requested and not self.resumed:
            state=runtime_control.snapshot(root/'train')
            task=state.get('tasks',{}).get('101',{})
            if task.get('status')=='paused':
                marker=root/'train/seeds/101/continuation/current.json'
                assert marker.is_file()
                events.append({'event':'paused','marker':json.loads(marker.read_text())})
                runtime_control.submit(root/'train','resume',task_id='101')
                self.resumed=True
probe=PauseOnce()
def build(entries,**kwargs):
    kwargs['callbacks'].append(probe)
    return original(entries,**kwargs)
tuning_ray.build_tuner=build
shared=['workload.orders=8','workload.initial_orders=4','workload.mean_interarrival=4',
'workload.base_ticks={A: 2,B: 2,C: 2,D: 2}','algorithm.epochs=1','experiment.run.seconds=120']
api.run_experiment(config,task='calibrate',sets=shared+[f'experiment.run.output="{root}/calibration"'])
result=api.run_experiment(config,task='train',tuning='auto',sets=shared+[
f'experiment.run.output="{root}/train"',f'experiment.run.calibration="{root}/calibration/calibration.json"',
'experiment.train_seeds=[101]','experiment.train_ticks=1000','experiment.run.probe_ticks=80',
'experiment.run.calibration_seconds=90','experiment.run.max_concurrent=1'])
assert result['status']=='completed',result
assert probe.resumed,events
assert result['seeds'][0]['physical_ticks']==1000,result
(root/'pause-proof.json').write_text(json.dumps(events,indent=2))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(ROOT / "configs/base-diagnostic/small.yaml"),
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
