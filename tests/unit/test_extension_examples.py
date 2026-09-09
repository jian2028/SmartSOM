"""Public example configuration and real custom encoder contracts."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_extensions import require_optional


def common(monkeypatch):
    require_optional("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    return importlib.import_module("extensions.common")


@pytest.mark.parametrize(
    "provider", ["sb3.maskable_ppo", "rllib.ppo", "rllib.resource_ppo"]
)
def test_example_prepares_pinned_public_recipe_and_real_distinct_networks(
    tmp_path, monkeypatch, provider
):
    torch = require_optional("torch")
    demo = common(monkeypatch)
    from smartsom.config.experiment import prepare
    from smartsom.learning.extensions import ObservationSpace
    from smartsom.learning.torch_extensions import build_actor_critic

    config = demo.build_config(provider, tmp_path / "evidence")
    prepared = prepare(config)
    algorithm = prepared.resolved.algorithm.algorithm
    assert algorithm.provider == provider
    assert prepared.resolved.run.budget.environment_steps == 128
    assert algorithm.parameters.n_steps == 32
    assert algorithm.extensions.network.actor.encoder.code_sha256
    assert algorithm.extensions.observation.name == "builtin.dict"
    assert not (tmp_path / "evidence").exists()
    space = ObservationSpace(fields=(("state", (3,)), ("local", (2,))))
    value = {"state": torch.ones(2, 3), "local": torch.zeros(2, 2)}
    role = "machine_policy" if provider == "rllib.resource_ppo" else None
    model = build_actor_critic(
        space, 4, algorithm.extensions.network, provider, role=role
    )
    result = model(value)
    assert result["logits"].shape == (2, 4) and result["values"].shape == (2,)
    assert isinstance(model.actor.encoder, demo.DemoEncoder)
    assert isinstance(model.critic.encoder, demo.DemoEncoder)
    assert model.actor.encoder.output_size != model.critic.encoder.output_size
    assert not {id(p) for p in model.actor.parameters()} & {
        id(p) for p in model.critic.parameters()
    }
    if role:
        other = build_actor_critic(
            space, 4, algorithm.extensions.network, provider, role="agv_policy"
        )
        assert model.actor.encoder.output_size != other.actor.encoder.output_size


@pytest.mark.parametrize("name", ["sb3", "rllib", "resource"])
def test_example_help_is_executable_and_does_not_start_a_run(tmp_path, name):
    require_optional("torch")
    import smartsom

    path = Path(__file__).resolve().parents[2] / "examples/extensions" / f"{name}.py"
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(smartsom.__file__).resolve().parents[1]),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint-source" in result.stdout
    assert not tuple(tmp_path.iterdir())
