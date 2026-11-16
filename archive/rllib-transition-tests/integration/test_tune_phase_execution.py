"""Real scientific phases and nested workers on the official Tune controller.

Small workloads establish correctness only; they are not performance evidence.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json
import sys
from pathlib import Path
import yaml
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation import allocate
from smartsom.experiments.tune_execution import fit_cohort
root, repository = Path(sys.argv[1]), Path(sys.argv[2])
method, workers = sys.argv[3], int(sys.argv[4])
episodes = 4 if workers else 2
nodes = [2, 4] if workers else [1, 2]
authored = root / 'experiment.yaml'
authored.write_text(yaml.safe_dump({
    'schema':'smartsom.base-experiment/v1',
    'scenario':str(repository/'configs/scenarios/small.yaml'),
    'algorithm':str(repository/f'configs/algorithms/base-{method}.yaml'),
}))
sets = [
    'experiment.run.task=train', 'experiment.run.seconds=null',
    'experiment.tracking.mode=disabled', 'experiment.drain_frozen=true',
    'experiment.drain_ticks=50', f'experiment.train_episodes={episodes}',
    'experiment.train_ticks=10000',
    f'experiment.periodic_evaluation.milestones={nodes}',
    'experiment.periodic_evaluation_cases=4',
    'experiment.periodic_evaluation.sample_action_seeds=1',
    'experiment.concurrency.evaluation_processes=2',
    'experiment.concurrency.evaluation_cpus=2',
    f'experiment.sampling.num_envs={workers or 1}',
    f'experiment.sampling.processes={workers}',
    'scenario.jobs.orders.total=8', 'scenario.jobs.orders.initial=4',
    'scenario.jobs.arrivals.mean_interval_ticks=4',
    'scenario.jobs.processing.base_ticks={A: 2,B: 2,C: 2,D: 2}',
    f'algorithm.production.policies.{method}.learner.params.epochs=1',
]
plan = compile_reservation(authored, sets=sets)
allocate(plan, root=root/'native')
results = fit_cohort([root/'native'], storage=root/'tune/phases',
                     resources={'cpus':3}, max_concurrent=1)
assert not results.errors, results.errors
rows = [json.loads(line) for line in (root/'native/episodes.jsonl').read_text().splitlines()]
assert len(rows) == episodes, rows
assert [row['episode'] for row in rows] == list(range(1,episodes+1))
assert rows[-1]['updates'] > 0
assert results[0].metrics['done']
for node in nodes:
    for mode in ('sample','argmax'):
        panel = root/f'native/periodic_evaluation/episode-{node:08d}'/mode
        assert (panel/'summary.json').is_file()
        cases = [json.loads(line) for line in (panel/'cases.jsonl').read_text().splitlines()]
        assert len(cases) == 4
        assert {row['condition'] for row in cases} == {'normal','low','medium','high'}
assert (root/'native/continuation/current.json').is_file()
assert (root/'native/last.pt').is_file()
events = [json.loads(p.read_text()) for p in (root/'native/control/tune-provenance').glob('*/*.json')]
starts = {e['attempt_id'] for e in events if e['event'] == 'actor_started'}
assert results[0].metrics['execution_attempt'] in starts
restores = [e for e in events if e['event'] == 'checkpoint_restored']
if not workers:
    assert restores and len(starts) > 1
for event in restores:
    assert event['attempt_id'] in starts and event['source_attempt'] in starts
    assert len(event['source_checkpoint_digest']) == 64
assert not any(e['event'] == 'execution_error' for e in events)
"""


@pytest.mark.parametrize("method,workers", [("ppo", 0), ("dqn", 2)])
def test_real_training_and_periodic_phases(tmp_path, method, workers):
    pytest.importorskip("ray")
    env = dict(
        os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_USAGE_STATS_ENABLED="0"
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), str(ROOT), method, str(workers)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_official_optuna_censoring_and_public_state_restore(tmp_path):
    import optuna
    from ray.tune.search.optuna import OptunaSearch

    from smartsom.experiments.tune_search_adapter import CensoredEvaluationSearch

    storage = optuna.storages.InMemoryStorage()

    def search():
        return CensoredEvaluationSearch(
            OptunaSearch(
                space={"x": optuna.distributions.FloatDistribution(0, 1)},
                metric=["time", "delay", "quality"],
                mode=["min", "min", "max"],
                seed=42,
                study_name="censor-test",
                storage=storage,
            )
        )

    original = search()
    original.suggest("complete")
    original.on_trial_result("complete", {"phase": "bootstrap"})
    original.on_trial_complete("complete", {"time": 10, "delay": 2, "quality": 0.9})
    original.suggest("censored")
    original.on_trial_complete(
        "censored",
        {
            "time": None,
            "delay": None,
            "quality": None,
            "search_evaluation/status": "censored",
        },
    )
    original.suggest("error")
    original.on_trial_complete(
        "error", {"time": 9, "delay": 1, "quality": 0.95}, error=True
    )
    study = optuna.load_study(study_name="censor-test", storage=storage)
    assert [t.state.name for t in study.trials] == ["COMPLETE", "PRUNED", "FAIL"]
    assert [t.number for t in study.best_trials] == [0]
    original.save(str(tmp_path / "search.pkl"))
    next_suggestion = original.suggest("next")
    restored = search()
    restored.restore(str(tmp_path / "search.pkl"))
    assert restored.suggest("next") == next_suggestion


def test_pareto_candidates_exclude_censored_and_dominated_results():
    from types import SimpleNamespace

    from smartsom.experiments.tune_search import pareto_candidates

    search = SimpleNamespace(
        objectives=[SimpleNamespace(mode=m) for m in ("min", "min", "max")]
    )

    def result(path, values):
        return SimpleNamespace(
            path=path,
            error=None,
            metrics={
                "done": True,
                **{f"search_objective_{i}": value for i, value in enumerate(values)},
            },
        )

    results = [
        result("A", [100, 10, 0.95]),
        result("B", [110, 8, 0.98]),
        result("C", [120, 12, 0.94]),
        result("D", [None, None, None]),
    ]
    assert pareto_candidates(results, search) == ["A", "B"]


def test_search_budget_before_first_panel_has_explicit_missing_evidence():
    from smartsom.config.search import Search
    from smartsom.experiments.tune_search import build_search
    from smartsom.experiments.tune_trainable import ExperimentTrainable

    settings = Search.model_validate(
        {
            "algorithm": "optuna",
            "num_samples": 1,
            "seed": 42,
            "train_seed": 101,
            "objectives": [
                {"panel": "sample", "metric": "configured.all.makespan", "mode": "min"}
            ],
            "parameters": {
                "algorithm.learning_rate": {
                    "distribution": "uniform",
                    "lower": 0.001,
                    "upper": 0.01,
                }
            },
        }
    )
    trainable = ExperimentTrainable.__new__(ExperimentTrainable)
    trainable.search = settings
    trainable.attempt_id = "test-attempt"
    trainable.recorded_run_id = "test-run"
    trainable.root = Path("/nonexistent-test-trial")
    trainable.state = {"phase": "complete"}
    report = trainable._report({"status": "wall_budget_reached"})
    assert report["done"] and report["search_objective_0"] is None
    assert report["search_evaluation"]["status"] == "not_evaluated"
    search = build_search(settings)
    search.suggest("budget")
    search.on_trial_complete(
        "budget",
        {
            "search_objective_0": None,
            "search_evaluation/status": "not_evaluated",
        },
    )


HPO_RESTORE_SCRIPT = (
    SCRIPT.split("plan = compile_reservation", 1)[0].replace(
        "experiment.tracking.mode=disabled", "experiment.tracking.mode=offline"
    )
    + r"""
from ray.tune import Callback
from smartsom.config.search import Search
from smartsom.experiments import control
import os
calibrated = sys.argv[5] == 'calibrated'
if calibrated:
    from smartsom.experiments.reservation import arguments
    from smartsom.experiments.reservation_runtime import calibrate
    calibration_plan = compile_reservation(authored, task='calibrate', sets=[
        *sets, 'experiment.tracking.mode=disabled',
        'experiment.run.seconds=120', 'experiment.run.calibration_cases=1',
    ])
    calibration_root = allocate(calibration_plan, root=root/'calibration')
    calibrate(arguments(calibration_root)[0])
    calibration_file = calibration_root/'calibration.json'
    original_calibration = calibration_file.read_bytes()
    assert json.loads(original_calibration)['status'] == 'frozen'
    sets += [f'experiment.run.calibration={json.dumps(str(calibration_file))}',
             'experiment.drain_frozen=false']
search = Search.model_validate({
    'algorithm':'optuna', 'num_samples':2, 'seed':42, 'train_seed':101,
    'objectives':[
        {'panel':'sample','metric':f'configured.all.{metric}','mode':mode}
        for metric,mode in [('makespan','min'),('mean_tardiness','min'),('final_passing_rate','max')]
    ],
    'parameters':{'algorithm.production.policies.ppo.learner.params.learning_rate':
                  {'distribution':'uniform','lower':0.0001,'upper':0.001}},
})
template = root/'template'
if calibrated:
    from smartsom.experiments.tune_hpo import compile_search, _freeze_search, _read_shared_calibration
    search_file = root/'search.yaml'
    search_file.write_text(yaml.safe_dump(search.model_dump(mode='json')))
    search_plan = compile_search(authored, search_file, sets=sets)
    allocate(search_plan.template, root=template)
    _freeze_search(template, search_plan)
    calibration_file.unlink()
else:
    plan = compile_reservation(authored, sets=sets)
    allocate(plan, root=template)
(root/'control').mkdir()
control.write_json(root/'control/owner.json', {
    'id':'search-integration-owner', 'root':str(root), 'status':'running',
    'owner':control.processes()[os.getpid()],
})
class PauseAfterPanel(Callback):
    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        import psutil
        with (root/'sdk-processes.jsonl').open('a') as stream:
            for process in psutil.Process().children(recursive=True):
                try:
                    if 'wandb' in process.name():
                        stream.write(json.dumps([process.pid,process.create_time()])+'\n')
                except psutil.NoSuchProcess:
                    pass
        if trial.last_result.get('search_evaluation') and not (root/'pause-once').exists():
            (root/'pause-once').write_text(trial.trial_id)
            control.request_pause(root)
results = fit_cohort([template], storage=root/'tune/search', resources={'cpus':3},
                     max_concurrent=1, search=search, callbacks=[PauseAfterPanel()])
assert (root/'pause-once').is_file()
assert any(not r.metrics.get('done') for r in results)
before = {p.parent.name:p.read_bytes() for p in (template/'trials').glob('*/suggestion.json')}
control.clear_pause(root)
results = fit_cohort([template], storage=root/'tune/search', resources={'cpus':3},
                     max_concurrent=1, search=search, resume=True)
assert not results.errors, results.errors
assert len(results) == 2 and all(r.metrics['done'] for r in results)
trials = list((template/'trials').iterdir())
assert len(trials) == 2
for trial in trials:
    if calibrated:
        derived = json.loads((trial/'inputs/search-calibration.json').read_text())
        retained, identity_inputs = _read_shared_calibration(trial, derived)
        assert retained == original_calibration
        args, settings = arguments(trial)
        assert settings.calibration is None
        from smartsom.config.base_diagnostic import load_inputs
        from smartsom.experiments.reservation_runtime import frozen_drain
        assert frozen_drain(args, load_inputs(args.experiment)[-1]) == derived['drain_ticks']
    if trial.name in before:
        assert (trial/'suggestion.json').read_bytes() == before[trial.name]
    rows = [json.loads(s) for s in (trial/'episodes.jsonl').read_text().splitlines()]
    assert [row['episode'] for row in rows] == [1,2]
    for node in (1,2):
        for mode in ('sample','argmax'):
            panel = trial/f'periodic_evaluation/episode-{node:08d}'/mode
            cases = [json.loads(s) for s in (panel/'cases.jsonl').read_text().splitlines()]
            assert len(cases) == 4
    journals = list((trial/'tracking').glob('*.jsonl'))
    assert journals
    events = [json.loads(s) for p in journals for s in p.read_text().splitlines()]
    committed = [event['event_id'] for event in events if event.get('event_id')]
    assert committed and len(committed) == len(set(committed))
    assert list((trial/'tracking/wandb').glob('offline-run-*'))
import psutil, time
observed = {tuple(json.loads(line)) for line in (root/'sdk-processes.jsonl').read_text().splitlines()}
assert observed, 'SDK process observation must be real, not vacuous'
def alive(identity):
    pid,birth = identity
    try:
        process = psutil.Process(pid)
        return process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
deadline = time.monotonic()+10
while any(map(alive,observed)) and time.monotonic() < deadline:
    time.sleep(0.1)
assert not any(map(alive,observed)), observed
import ray
assert not ray.is_initialized()
"""
)


@pytest.mark.parametrize("calibration", ["fixed", "calibrated"])
def test_interrupted_optuna_search_restores_with_offline_tracking(
    tmp_path, calibration
):
    pytest.importorskip("ray")
    pytest.importorskip("wandb")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            HPO_RESTORE_SCRIPT,
            str(tmp_path),
            str(ROOT),
            "ppo",
            "0",
            calibration,
        ],
        cwd=ROOT,
        env=dict(
            os.environ,
            RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
            RAY_USAGE_STATS_ENABLED="0",
            WANDB_MODE="offline",
        ),
        capture_output=True,
        text=True,
        timeout=240,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actor_cleanup_joins_tracking_service_even_when_session_close_fails(
    monkeypatch,
):
    from types import SimpleNamespace

    from smartsom.experiments.tune_trainable import ExperimentTrainable

    closed = []

    def fail():
        raise RuntimeError("session close failed")

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(teardown=lambda: closed.append(True))
    )
    actor = ExperimentTrainable.__new__(ExperimentTrainable)
    actor.session = SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError, match="session close failed"):
        actor.cleanup()
    assert closed == [True]


@pytest.mark.parametrize("corruption", ["history", "payload"])
def test_failed_actor_restore_preserves_science_and_records_no_success(
    tmp_path, corruption
):
    import json
    from types import SimpleNamespace

    from smartsom.experiments.tune_checkpoint import save
    from smartsom.experiments.tune_provenance import directory, record, snapshot
    from smartsom.experiments.tune_trainable import ExperimentTrainable

    root = tmp_path / "run"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({"run_id": "run"}))
    receipt = root / "episodes.jsonl"
    receipt.write_text("checkpoint evidence")
    record(root, "trial", "actor_started", attempt_id="first")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save(
        root,
        checkpoint,
        {"phase": "training"},
        execution=snapshot(root, "trial", "first"),
    )
    receipt.write_text("later committed evidence")
    record(root, "trial", "actor_started", attempt_id="second")
    if corruption == "history":
        event = next(directory(root, "trial").glob("*.json"))
        event.write_text("{}")
    else:
        (checkpoint / "native/episodes.jsonl").write_text("corrupt")
    actor = ExperimentTrainable.__new__(ExperimentTrainable)
    actor._trial_info = SimpleNamespace(trial_id="trial")
    actor.execution_config = {"root": str(root)}
    actor.root = root
    actor.attempt_id = "second"
    actor.immutable_digests = {}
    with pytest.raises(ValueError, match="digest mismatch"):
        actor.load_checkpoint(checkpoint)
    assert receipt.read_text() == "later committed evidence"
    assert all(
        json.loads(p.read_text()).get("event") != "checkpoint_restored"
        for p in directory(root, "trial").glob("*.json")
    )
