"""Official Tune cohorts release phase resources and restore committed work."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json, os, sys
from pathlib import Path
from ray.tune import Callback
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation import allocate
from smartsom.experiments.tune_execution import fit_cohort
from smartsom.experiments import control
root, repository = Path(sys.argv[1]), Path(sys.argv[2])
pause = sys.argv[3] == 'pause'
sets = [
    'experiment.run.task=train', 'experiment.run.seconds=null',
    'experiment.tracking.mode=disabled', 'experiment.drain_frozen=true',
    'experiment.drain_ticks=50', 'experiment.train_episodes=4',
    'experiment.train_ticks=10000',
    'experiment.periodic_evaluation.milestones=[2,4]',
    'experiment.periodic_evaluation_cases=4',
    'experiment.periodic_evaluation.sample_action_seeds=1',
    'experiment.concurrency.evaluation_processes=2',
    'experiment.concurrency.evaluation_cpus=2',
    'scenario.jobs.orders.total=8', 'scenario.jobs.orders.initial=4',
    'scenario.jobs.arrivals.mean_interval_ticks=4',
    'scenario.jobs.processing.base_ticks={A: 2,B: 2,C: 2,D: 2}',
    'algorithm.production.policies.ppo.learner.params.epochs=1',
]
roots = []
for seed in (101, 102):
    plan = compile_reservation(
        repository/'experiments/examples/small/default/experiment.yaml',
        seed=seed, sets=sets,
    )
    native = root/f'seed-{seed}'
    allocate(plan, root=native)
    roots.append(native)

class PauseOnce(Callback):
    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        if trial.last_result.get('episodes', 0) >= 1 and not (root/'pause-once').exists():
            (root/'pause-once').write_text('requested')
            control.request_pause(root)

callbacks = []
if pause:
    (root/'control').mkdir()
    control.write_json(root/'control/owner.json', {
        'id':'integration-owner', 'root':str(root), 'status':'running',
        'owner':control.processes()[os.getpid()],
    })
    callbacks = [PauseOnce()]
results = fit_cohort(roots, storage=root/'tune/cohort', resources={'cpus':3},
                     max_concurrent=2, callbacks=callbacks)
if pause:
    assert (root/'pause-once').exists()
    assert any(not result.metrics.get('done') for result in results)
    control.clear_pause(root)
    results = fit_cohort(roots, storage=root/'tune/cohort', resources={'cpus':3},
                         max_concurrent=2, resume=True)
assert not results.errors, results.errors
assert len(results) == 2 and all(result.metrics['done'] for result in results)
# Both trials need the whole three-CPU budget for evaluation. Completing both
# proves training actors do not hold resources while awaiting their panels.
for native in roots:
    rows = [json.loads(line) for line in (native/'episodes.jsonl').read_text().splitlines()]
    assert [row['episode'] for row in rows] == [1,2,3,4]
    assert rows[-1]['updates'] > 0
    for node in (2,4):
        for mode in ('sample','argmax'):
            panel = native/f'periodic_evaluation/episode-{node:08d}'/mode
            assert (panel/'summary.json').is_file()
            cases = [json.loads(line) for line in (panel/'cases.jsonl').read_text().splitlines()]
            assert len(cases) == 4
    assert json.loads((native/'tune-status.json').read_text())['status'] == 'completed'
# Finished restoration must not train again or republish evaluation evidence.
before = {str(p): p.read_bytes() for native in roots for p in native.rglob('*.jsonl')}
restored = fit_cohort(roots, storage=root/'tune/cohort', resources={'cpus':3}, resume=True)
assert len(restored) == 2 and not restored.errors
assert all(Path(name).read_bytes() == data for name, data in before.items())
import ray
assert not ray.is_initialized()
"""


@pytest.mark.parametrize("mode", ["cohort", "pause"])
def test_official_cohort_and_whole_run_restore(tmp_path, mode):
    pytest.importorskip("ray")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), str(ROOT), mode],
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
import yaml
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation import allocate
from smartsom.experiments.tune_execution import fit_cohort
from smartsom.experiments import tune_trainable
root, repository = Path(sys.argv[1]), Path(sys.argv[2])
limit = int(sys.argv[3])
exhaust = sys.argv[4] == 'exhaust'
method = sys.argv[5]
document = yaml.safe_load((repository/'experiments/examples/small/default/experiment.yaml').read_text())
document['scenario'] = str(repository/'configs/scenarios/small.yaml')
document['algorithm'] = str(repository/f'configs/algorithms/base-{method}.yaml')
authored = root/'experiment.yaml'
authored.write_text(yaml.safe_dump(document))
plan = compile_reservation(authored, sets=[
    'experiment.run.task=train', 'experiment.run.seconds=null',
    'experiment.tracking.mode=disabled', 'experiment.drain_frozen=true',
    'experiment.drain_ticks=30', 'experiment.train_episodes=2',
    'experiment.periodic_evaluation.milestones=[]', 'experiment.final_evaluation=false',
    'scenario.jobs.orders.total=8', 'scenario.jobs.orders.initial=4',
])
native = root/'native'
allocate(plan, root=native)
# This subclass is serialized to the real Ray actor. The fault is raised after
# Tune has checkpointed the first episode, not via a parent-process monkeypatch.
class FailOnce(tune_trainable.ExperimentTrainable):
    def _session(self):
        fresh = self.session is None
        session = super()._session()
        if method == 'dqn' and fresh and (root/'injected-failure').exists():
            import torch
            from smartsom.experiments.reservation_session import tuples
            from tests.integration.test_rule_learning import _equal_dqn_values
            expected = torch.load(root/'before-failure.pt', map_location='cpu', weights_only=False)
            actual = session.policy.get_state()
            _equal_dqn_values(expected['policy'], actual)
            assert session.rng.getstate() == tuples(expected['condition_rng'])
            learner = actual['learner']
            assert learner['updates'] > 0
            assert any(value['state'] for value in learner['optimizers'].values())
            assert any(len(rows) for rows in learner['replay'].values())
            assert self.attempt_id != expected['attempt']
            (root/'verified-dqn-restore.json').write_text(json.dumps({
                'attempt':self.attempt_id, 'state_equal':True,
                'condition_rng_equal':True, 'optimizer_nonempty':True, 'replay_nonempty':True,
            }))
        return session

    def step(self):
        receipt = self.root/'episodes.jsonl'
        marker = root/'injected-failure'
        if receipt.exists() and len(receipt.read_text().splitlines()) == 1 and (exhaust or not marker.exists()):
            if method == 'dqn':
                import torch
                pointer = json.loads((self.root/'continuation/current.json').read_text())
                generation = self.root/'continuation'/pointer['generation']
                saved = json.loads((generation/'state.json').read_text())
                policy = torch.load(generation/'model.pt', map_location='cpu', weights_only=False)['state']
                torch.save({'policy':policy, 'condition_rng':saved['condition_rng'],
                            'attempt':self.attempt_id}, root/'before-failure.pt')
            (root/'first-row.json').write_text(receipt.read_text())
            with (root/'failures.jsonl').open('a') as stream:
                stream.write(json.dumps({'attempt': self.attempt_id})+'\n')
            marker.write_text(self.trial_id)
            raise RuntimeError('injected remote failure after committed episode')
        return super().step()
tune_trainable.ExperimentTrainable = FailOnce
options = dict(storage=root/'tune/retry', resources={'cpus':1})
if limit:
    options['max_failures'] = limit
results = fit_cohort([native], **options)
assert (root/'injected-failure').exists()
trial_id = (root/'injected-failure').read_text()
if exhaust:
    assert not sys.modules['ray'].is_initialized()
    assert len(results.errors) == 1
    assert len((root/'failures.jsonl').read_text().splitlines()) == 2
    assert len((native/'episodes.jsonl').read_text().splitlines()) == 1
    sys.exit(0)
if not limit:
    assert len(results.errors) == 1
    assert len((native/'episodes.jsonl').read_text().splitlines()) == 1
    untouched = (native/'episodes.jsonl').read_bytes()
    again = fit_cohort([native], **options, resume=True)
    assert len(again.errors) == 1
    assert (native/'episodes.jsonl').read_bytes() == untouched
    results = fit_cohort([native], **options, resume=True, retry_failed=True)
assert not results.errors, results.errors
assert results[0].metrics['done']
assert results[0].metrics['trial_id'] == trial_id
rows = [json.loads(line) for line in (native/'episodes.jsonl').read_text().splitlines()]
assert [row['episode'] for row in rows] == [1,2]
assert rows[1]['updates'] > rows[0]['updates']
assert rows[0] == json.loads((root/'first-row.json').read_text())
if method == 'dqn':
    verified = json.loads((root/'verified-dqn-restore.json').read_text())
    assert all(verified[key] for key in ('state_equal','condition_rng_equal','optimizer_nonempty','replay_nonempty'))
events = [json.loads(p.read_text()) for p in (native/'control/tune-provenance'/trial_id).glob('*.json')]
assert any(e['event'] == 'checkpoint_restored' for e in events)
import ray
assert not ray.is_initialized()
"""


@pytest.mark.parametrize(
    "max_failures,mode,method",
    [(0, "once", "ppo"), (1, "once", "ppo"), (1, "exhaust", "ppo"), (0, "once", "dqn")],
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
