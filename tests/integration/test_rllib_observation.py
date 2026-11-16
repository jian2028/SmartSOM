"""Lossless fixed-space encoding of real production and waiting states."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from smartsom.config.base_diagnostic import BaseAlgorithm
from smartsom.learning.factory_episode import FactoryEpisode
from smartsom.learning.rllib_observation import SUPPLY_FIELDS, V2ObservationCodec
from smartsom.representations.v2.network import pack
from tests.unit.test_base_reservations import sim_with


@pytest.mark.parametrize(
    "values",
    [
        (None,),
        ("1.5",),
        (True, False),
        (2**54 + 2**30 + 1,),
        (2**63, 0),
        (2**70, 0.1),
        (2**63, -1),
        (float("inf"), -0.0),
    ],
)
def test_numpy_numeric_fields_keep_torch_conversion_boundaries(values):
    import torch

    from smartsom.representations.v2._packing import NumpyArrays

    try:
        expected = torch.tensor(values).to(torch.float32).numpy()
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        with pytest.raises(type(error)):
            NumpyArrays.tensor(values)
    else:
        # Candidate/peer destinations are float32 in both packing backends.
        actual = np.asarray(NumpyArrays.tensor(values), dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("limit", [3, 1000])
@pytest.mark.parametrize("fragment_length", [1, 7])
def test_event_waits_preserve_cost_and_official_pending_agent_rewards(
    limit, fragment_length
):
    from copy import deepcopy

    import gymnasium as gym
    from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

    from smartsom.learning.rllib_dqn import (
        RestorableMultiAgentPrioritizedEpisodeReplayBuffer,
    )
    from smartsom.learning.rllib_env import FactoryMultiAgentEnv
    from smartsom.rewards.normalized_task import cost_snapshot
    from tests.integration.test_rllib_dqn_checkpoint import assert_state_equal

    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/rllib-base-dqn.yaml").read_text())
    )
    env = FactoryMultiAgentEnv(
        {
            "scenario": replace(sim_with(n=2).scenario, tick_limit=limit),
            "options": options,
            "algorithm": "dqn",
        }
    )
    try:
        observations, infos = env.reset(seed=101)
        trajectory = MultiAgentEpisode(
            observation_space=gym.spaces.Dict(env.observation_spaces),
            action_space=gym.spaces.Dict(env.action_spaces),
        )
        trajectory.add_env_reset(observations=observations, infos=infos)
        fragment = deepcopy(trajectory)
        replay = RestorableMultiAgentPrioritizedEpisodeReplayBuffer(capacity=10000)
        totals, jumps = {}, []
        observed_reward = 0.0
        for _ in range(10000):
            started = set(env.started)
            before = cost_snapshot(env.episode.sim)
            tick = env.episode.sim.tick
            actions = {
                agent: int(np.flatnonzero(obs["action_mask"])[0])
                for agent, obs in observations.items()
            }
            observations, rewards, terminated, truncated, infos = env.step(actions)
            delta = cost_snapshot(env.episode.sim).reward_since(before)
            assert set(rewards) == started
            assert all(value == pytest.approx(delta) for value in rewards.values())
            observed_reward += delta
            jumps.append(env.episode.sim.tick - tick)
            for agent, value in rewards.items():
                totals[agent] = totals.get(agent, 0.0) + value
            transition = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                infos=infos,
                terminateds=terminated,
                truncateds=truncated,
                extra_model_outputs={agent: {} for agent in actions},
            )
            trajectory.add_env_step(**transition)
            fragment.add_env_step(**transition)
            done = terminated["__all__"] or truncated["__all__"]
            if trajectory.env_t % fragment_length == 0 or done:
                successor = None if done else fragment.cut(len_lookback_buffer=1)
                # Match the official EnvRunner's fragment admission preprocessing.
                MultiAgentEnvRunner._prune_zero_len_sa_episodes(fragment)
                replay.add(fragment)
                fragment = successor
            if done:
                break
        else:
            pytest.fail("event-driven episode did not reach its horizon")
        assert truncated["__all__"] == (limit == 3)
        assert trajectory.is_terminated == (limit == 1000)
        assert trajectory.is_truncated == (limit == 3)
        assert set(observations) == set(totals)
        assert all(
            np.flatnonzero(obs["action_mask"]).tolist() == [0]
            for obs in observations.values()
        )
        for agent, total in totals.items():
            episode = trajectory.agent_episodes[agent]
            assert sum(episode.get_rewards()) == pytest.approx(total)
            assert episode.is_terminated == (limit == 1000)
            assert episode.is_truncated == (limit == 3)
        restored = RestorableMultiAgentPrioritizedEpisodeReplayBuffer(capacity=10000)
        restored.set_state(replay.get_state())
        for buffer in (replay, restored):
            combined = buffer.episodes[0]
            assert combined.is_terminated == (limit == 1000)
            assert combined.is_truncated == (limit == 3)
            for agent, expected in trajectory.agent_episodes.items():
                actual = combined.agent_episodes[agent]
                assert actual.is_terminated == expected.is_terminated
                assert actual.is_truncated == expected.is_truncated
                for method in ("get_rewards", "get_actions", "get_observations"):
                    assert_state_equal(
                        getattr(actual, method)(), getattr(expected, method)()
                    )
        assert observed_reward == pytest.approx(env.episode.team_return)
        if limit == 1000:
            # Real physics intervals coalesce, costs do not vanish.
            assert max(jumps) > 1
            assert observed_reward == pytest.approx(
                -env.episode.metrics()["joint_cost"]
            )
    finally:
        env.close()


def test_rule_action_spaces_do_not_scale_with_physical_job_pairs():
    from smartsom.learning.rllib_env import FactoryMultiAgentEnv

    root = Path(__file__).resolve().parents[2]
    document = yaml.safe_load(
        (root / "configs/algorithms/rllib-base-dqn.yaml").read_text()
    )
    for name in ("observation", "network"):
        document["production"]["policies"]["dqn"][name]["params"] = {
            "social": {"kind": "implicit", "fusion": "attention"}
        }
    options = BaseAlgorithm.model_validate(document)
    widths = []
    for jobs in (2, 128):
        env = FactoryMultiAgentEnv(
            {
                "scenario": sim_with(n=jobs).scenario,
                "options": options,
                "algorithm": "dqn",
            }
        )
        try:
            widths.append({group: codec.width for group, codec in env.codecs.items()})
            if jobs == 128:
                assert env.codecs["post"].shapes["accept"]["jobs"][0] == 5
                assert env.codecs["input"].shapes["accept"]["jobs"][0] == 128
        finally:
            env.close()
    assert widths[0] == widths[1]
    # Seven process rules, three machine modes; two reserved specials and WAIT.
    assert widths[1]["machine"] == 24
    # Two stages share one input module: at most six rules plus specials/WAIT.
    assert widths[1]["input"] == 9


@pytest.mark.parametrize("fusion", ["base", "attention", "concat"])
def test_real_observations_and_waits_preserve_payload(fusion, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    social = (
        {"kind": "none"} if fusion == "base" else {"kind": "implicit", "fusion": fusion}
    )
    if fusion == "concat":
        social["slots"] = {
            "stations": 8,
            "vehicles": 8,
            "input_jobs": 2,
            "post_jobs": 2,
        }
    policy = doc["production"]["policies"]["dqn"]
    policy["observation"]["params"] = {"social": social}
    policy["network"]["params"] = {"social": social}
    options = BaseAlgorithm.model_validate(doc)
    episode = FactoryEpisode(sim_with(n=2).scenario, options)
    codecs = {}
    examples = {}
    checks = waits = 0
    checked_distinct_supply = False

    def codec(request):
        group = request.group + (
            "." + request.stage if request.group == "dispatcher" else ""
        )
        if group not in codecs:
            codecs[group] = V2ObservationCodec(
                group,
                options=options,
                actions=32,
                stations=8,
                vehicles=8,
                jobs=2,
                social=social,
            )
        return codecs[group]

    while True:
        frame = episode.frame
        if frame.request is not None:
            request = frame.request
            if fusion != "base":
                from smartsom.domain.public_facts import freeze
                from smartsom.engine.decision_facts import DecisionFacts
                from smartsom.representations.v2.facts import view
                from smartsom.representations.v2.observation import social_relations

                # Independently rebuild every candidate's relation from fresh
                # public state, including successive decisions at the same tick.
                with (
                    episode.protocol.local_facts(),
                    DecisionFacts(episode.protocol, request) as facts,
                ):
                    expected = tuple(
                        freeze(
                            social_relations(
                                view(facts),
                                request,
                                candidate,
                                ordered=fusion == "concat",
                            )
                        )
                        for candidate in request.candidates
                    )
                assert request.inputs["social"] == expected
            encoder = codec(request)
            if checks == 0:
                import torch

                def forbidden_tensor_allocation(*args, **kwargs):
                    pytest.fail("numeric environment encoding allocated a Torch tensor")

                with monkeypatch.context() as patch:
                    patch.setattr(torch, "tensor", forbidden_tensor_allocation)
                    patch.setattr(torch, "zeros", forbidden_tensor_allocation)
                    obs = encoder.encode(request)
            else:
                obs = encoder.encode(request)
            if (
                not checked_distinct_supply
                and fusion != "base"
                and request.group in {"input", "post"}
                and request.stage == "accept"
                and len(request.inputs["social"]) > 1
                and request.inputs["social"][1]["jobs"]
            ):
                relations = list(request.inputs["social"])
                relation = dict(relations[1])
                jobs = list(relation["jobs"])
                job = dict(jobs[0])
                values = job["condition"]
                job["condition"] = (values[0] + 1, *values[1:])
                jobs[0] = job
                relation["jobs"] = tuple(jobs)
                relations[1] = relation
                with pytest.raises(ValueError, match="supply differs"):
                    encoder.encode(
                        replace(
                            request,
                            inputs={**request.inputs, "social": tuple(relations)},
                        )
                    )
                checked_distinct_supply = True
            examples.setdefault(encoder.group, {})["production"] = obs
            if checks == 0:
                for invalid in (None, "1.5"):
                    inputs = dict(request.inputs)
                    inputs["context"] = (invalid, *inputs["context"][1:])
                    with pytest.raises((TypeError, ValueError)):
                        encoder.encode(replace(request, inputs=inputs))
                for bad in (
                    replace(request, group="wrong"),
                    replace(request, stage="wrong"),
                ):
                    with pytest.raises(ValueError, match="role/stage"):
                        encoder.encode(bad)
                    with pytest.raises(ValueError, match="role/stage"):
                        encoder.decode(bad, 1)
                wrong_social = (
                    {
                        "kind": "implicit",
                        "fusion": "concat",
                        "slots": {
                            "stations": 8,
                            "vehicles": 8,
                            "input_jobs": 2,
                            "post_jobs": 2,
                        },
                    }
                    if fusion != "concat"
                    else {"kind": "implicit", "fusion": "attention"}
                )
                with pytest.raises(ValueError, match="configuration mismatch"):
                    V2ObservationCodec(
                        encoder.group,
                        options=options,
                        actions=32,
                        stations=8,
                        vehicles=8,
                        jobs=2,
                        social=wrong_social,
                    )
                if fusion != "base":
                    with pytest.raises(ValueError, match="social schema"):
                        encoder.encode(
                            replace(request, inputs={**request.inputs, "social": ()})
                        )
            assert encoder.observation_space.contains(obs)
            assert not obs["action_mask"][0]
            assert obs["action_mask"].sum() == len(request.candidates)
            old = pack([request], critics=None, options=options)["stages"][
                request.stage
            ]
            actual = obs["stages"][request.stage]
            for key, tensor in old.items():
                if key == "indices":
                    continue
                expected = tensor[0].numpy()
                target = (
                    np.broadcast_to(
                        actual[key], (expected.shape[0], *actual[key].shape)
                    )
                    if key in SUPPLY_FIELDS
                    else actual[key]
                    if key == "context"
                    else actual[key][1:]
                )
                np.testing.assert_array_equal(
                    target[tuple(slice(0, n) for n in expected.shape)], expected
                )
            for index, candidate in enumerate(request.candidates, 1):
                assert encoder.decode(request, index) == candidate.identity
            with pytest.raises(ValueError):
                encoder.decode(request, 0)
            checks += 1
            episode.act(encoder.decode(request, 1))
        else:
            for context in frame.waiting:
                encoder = codec(context)
                obs = encoder.encode_wait(episode.protocol, context)
                examples.setdefault(encoder.group, {})["waiting"] = obs
                with pytest.raises(ValueError, match="role/stage"):
                    encoder.encode_wait(
                        episode.protocol, replace(context, group="wrong")
                    )
                assert encoder.observation_space.contains(obs)
                assert obs["waiting"].tolist() == [1]
                assert np.flatnonzero(obs["action_mask"]).tolist() == [0]
                waits += 1
            if frame.terminated or frame.truncated:
                break
            episode.wait()
    assert checks > 0 and waits > checks
    assert fusion == "base" or checked_distinct_supply
    episode.close()
    verify_official_modules(codecs, examples, social)


def verify_official_modules(codecs, examples, social):
    from dataclasses import asdict

    import gymnasium as gym
    import torch
    import tree
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
        DefaultDQNTorchRLModule,
    )
    from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    from smartsom.learning.rllib_models import ResearchDQNCatalog, ResearchPPOCatalog

    torch.set_num_threads(1)
    for group, rows in examples.items():
        codec = codecs[group]
        obs = tree.map_structure(
            lambda *values: torch.as_tensor(np.stack(values)),
            rows["production"],
            rows["waiting"],
        )
        config = asdict(DefaultModelConfig()) | dict(DQNConfig().model_config)
        config.update(
            research_group=group, research_social=social, research_stages=codec.stages
        )
        torch.manual_seed(101)
        dqn = DefaultDQNTorchRLModule(
            observation_space=codec.observation_space,
            action_space=codec.action_space,
            model_config=config,
            catalog_class=ResearchDQNCatalog,
        )
        dqn.make_target_networks()
        batch = {Columns.OBS: obs, Columns.NEXT_OBS: obs}
        mask = obs["action_mask"].bool()
        result = dqn.forward_train(batch)
        for key in ("qf_preds", "qf_next_preds", "qf_target_next_preds"):
            assert torch.isfinite(result[key][mask]).all()
            assert torch.isneginf(result[key][~mask]).all()
        # A numerical loss fixture, not a change to the factory's reward.
        # Truncation retains bootstrap; only true termination removes it.
        actions = mask.long().argmax(-1)
        loss_batch = {
            Columns.ACTIONS: actions,
            Columns.REWARDS: torch.full((2,), 0.37),
            Columns.TERMINATEDS: torch.zeros(2, dtype=torch.bool),
            Columns.TRUNCATEDS: torch.ones(2, dtype=torch.bool),
            "n_step": torch.ones(2),
            "weights": torch.ones(2),
        }
        loss_config = DQNConfig()
        loss_learner = DQNTorchLearner(config=loss_config, module=dqn)
        actual_loss = loss_learner.compute_loss_for_module(
            module_id="default_policy",
            config=loss_config,
            batch=loss_batch,
            fwd_out=result,
        )
        next_values = []
        for row in range(2):
            legal = torch.where(mask[row])[0]
            greedy = legal[result["qf_next_preds"][row, legal].argmax()]
            next_values.append(result["qf_target_next_preds"][row, greedy])
        target = (
            loss_batch[Columns.REWARDS] + loss_config.gamma * torch.stack(next_values)
        ).detach()
        predicted = result["qf_preds"].gather(1, actions[:, None]).squeeze(1)
        assert not torch.isclose(
            torch.nn.functional.huber_loss(predicted, target),
            torch.nn.functional.huber_loss(predicted, loss_batch[Columns.REWARDS]),
        ), "bootstrap oracle must distinguish truncation from termination"
        torch.testing.assert_close(
            actual_loss, torch.nn.functional.huber_loss(predicted, target)
        )
        for timestep in (0, 1000000):
            for _ in range(8):
                actions = dqn.forward_exploration(batch, t=timestep)[Columns.ACTIONS]
                assert mask.gather(1, actions[:, None]).all()
        ppo_space = gym.spaces.Dict(
            {
                **codec.observation_space.spaces,
                "critic": gym.spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32),
            }
        )
        ppo = DefaultPPOTorchRLModule(
            observation_space=ppo_space,
            action_space=codec.action_space,
            model_config=config,
            catalog_class=ResearchPPOCatalog,
        )
        ppo_obs = {**obs, "critic": torch.zeros(2, 48)}
        first = ppo.forward_train({Columns.OBS: ppo_obs})[Columns.ACTION_DIST_INPUTS]
        second = ppo.forward_train(
            {Columns.OBS: {**ppo_obs, "critic": torch.ones(2, 48)}}
        )[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(first, second)
        assert torch.isfinite(first[mask]).all() and torch.isneginf(first[~mask]).all()
        values = ppo.compute_values({Columns.OBS: ppo_obs})
        assert torch.isfinite(values).all()
        assert not torch.equal(
            values,
            ppo.compute_values({Columns.OBS: {**ppo_obs, "critic": torch.ones(2, 48)}}),
        )


@pytest.mark.parametrize("tick_limit", [12, 1000])
@pytest.mark.parametrize("subject_dependent", [False, True])
def test_critic_reuse_preserves_every_frame_and_observation_ownership(
    tick_limit, subject_dependent
):
    from smartsom.learning.rllib_env import FactoryMultiAgentEnv

    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/rllib-base-ppo.yaml").read_text())
    )
    scenario = replace(sim_with(n=2).scenario, tick_limit=tick_limit)
    envs = [
        FactoryMultiAgentEnv(
            {"scenario": scenario, "options": options, "algorithm": "ppo"}
        )
        for _ in range(2)
    ]
    calls = [0, 0]

    def counted(critic, index):
        def observe(facts, context):
            calls[index] += 1
            values = critic(facts, context)
            if subject_dependent:
                # Explicit custom component with context-dependent output.
                values = (float(sum(map(ord, context.owner))), *values[1:])
            return values

        if index == 1 and not subject_dependent:
            observe.subject_independent = True
        return observe

    def equal(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                equal(a, b)
        else:
            np.testing.assert_array_equal(left, right)

    multi_waits = same_tick_changes = 0
    try:
        rows = [env.reset(seed=101) for env in envs]
        for index, env in enumerate(envs):
            env.episode.components.critic = counted(
                env.episode.components.critic, index
            )
        for _ in range(3000):
            equal(*rows)
            obs = rows[0][0]
            candidate = rows[1][0]
            if len(candidate) > 1:
                multi_waits += 1
                arrays = [value["critic"] for value in candidate.values()]
                saved = [a.copy() for a in arrays]
                arrays[0][:] = -12345
                for actual, expected in zip(arrays[1:], saved[1:], strict=True):
                    np.testing.assert_array_equal(actual, expected)
                arrays[0][:] = saved[0]
            if envs[0].episode.frame.terminated or envs[0].episode.frame.truncated:
                break
            actions = {
                agent: int(np.flatnonzero(value["action_mask"])[0])
                for agent, value in obs.items()
            }
            previous_tick = envs[0].episode.sim.tick
            rows = [env.step(actions) for env in envs]
            if envs[0].episode.sim.tick == previous_tick:
                same_tick_changes += 1
            # Returned observations remain detached from subsequent frames.
            equal(obs, candidate)
        else:
            pytest.fail("representative factory episode did not reach its boundary")
        assert multi_waits > 0 and same_tick_changes > 0
        if subject_dependent:
            assert calls[0] == calls[1]
        else:
            assert calls[0] > calls[1] > 0
        assert envs[0].episode.frame.truncated == (tick_limit == 12)
        # A second reset must build fresh features and match the original reset.
        equal(envs[0].reset(seed=101), envs[1].reset(seed=101))
    finally:
        for env in envs:
            env.close()
