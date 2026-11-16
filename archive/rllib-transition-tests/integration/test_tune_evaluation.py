"""Actual Ray case replenishment, ordered publication and failure propagation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = r"""
import json
import os
import sys
import time
from pathlib import Path
import ray
from ray import tune
from ray.tune.execution.placement_groups import PlacementGroupFactory
from smartsom.experiments import parallel_evaluation
root = Path(sys.argv[1])
crash = sys.argv[2] == 'crash'

def case(settings, task):
    index = task['index']
    path = Path(settings['output']) / 'events.jsonl'
    def record(kind):
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps({'kind':kind,'index':index,'time':time.monotonic()})+'\n').encode())
        finally:
            os.close(fd)
    record('start')
    if crash and index == 1:
        os._exit(7)
    time.sleep(2 if index == 0 else .05)
    record('end')
    return index

class Panel(tune.Trainable):
    def step(self):
        parallel_evaluation._case = case
        rows = list(parallel_evaluation.case_results(
            {'output':str(root)}, [{'index':i} for i in range(5)], None,
            workers=2, on_progress=lambda *_: None))
        return {'done':True,'rows':rows}

ray.init(address='local',num_cpus=3,include_dashboard=False,log_to_driver=False,
         object_store_memory=80*1024**2)
try:
    results = tune.Tuner(tune.with_resources(Panel,PlacementGroupFactory([
        {'CPU':1},{'CPU':1},{'CPU':1}])),
        run_config=tune.RunConfig(storage_path=str(root),name='tune',verbose=0,
            failure_config=tune.FailureConfig(max_failures=0),
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False))).fit()
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    if crash:
        assert len(results.errors) == 1
        assert sum(e['kind']=='start' and e['index']==1 for e in events)==1
        assert not results[0].metrics.get('rows')
    else:
        assert not results.errors
        assert results[0].metrics['rows'] == [(i,i) for i in range(5)]
        started = {e['index']:e['time'] for e in events if e['kind']=='start'}
        ended = {e['index']:e['time'] for e in events if e['kind']=='end'}
        assert started[2] < ended[0], events
finally:
    ray.shutdown()
"""


@pytest.mark.parametrize("mode", ["normal", "crash"])
def test_ray_cases_replenish_and_fail_without_application_retry(tmp_path, mode):
    env = dict(
        os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_USAGE_STATS_ENABLED="0"
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), mode],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    (tmp_path / "driver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
