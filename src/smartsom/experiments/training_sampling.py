"""Synchronous process sampling used by the existing reservation session.

Each worker owns its simulator and episode-local component state. Only the parent
updates networks. A wave is committed after all assigned episodes finish; an
interrupted wave replays from its last durable boundary, as in serial training.
"""

import multiprocessing
import pickle
import time
import traceback


def _send(connection, value):
    # Bytes keep Torch's multiprocessing reducer from creating external shared
    # storage lifetimes or file-descriptor dependencies across stop/restore.
    connection.send_bytes(pickle.dumps(value, protocol=5))


def _receive(connection):
    return pickle.loads(connection.recv_bytes())


def _worker(connection, options, seed, index):
    import torch

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import seed_for
    from smartsom.experiments.reservation_runtime import episode
    from smartsom.learning.sampling import take_rollout

    try:
        policy = build_policy(options, seed, numerical_threads=1)
        policy.seed_sampling(seed_for(seed, "sampler", index))
        while True:
            command = _receive(connection)
            if command is None:
                return
            weights, state, arguments, keywords, fragments = command
            policy.learner.load_sampling_state(weights)
            if state:
                policy.learner.generator.set_state(state["generator"])
                torch.set_rng_state(state["torch_rng"])
                policy.training_components.reward_state = state["reward"]
                for name, value in state["policies"].items():
                    policy.policies[name].set_state(value)

            def progress(value):
                _send(connection, ("progress", value))

            def fragment(bound, rewards, critic):
                packet = take_rollout(
                    bound, rewards, critic, terminal=False, episode_end=False
                )
                _send(connection, ("fragment", packet))
                weights = _receive(connection)
                policy.learner.load_sampling_state(weights)

            factory, workload, world_seed, condition, drain = arguments
            report, rewards, critic = episode(
                factory,
                workload,
                options,
                policy,
                world_seed,
                condition,
                drain,
                on_fragment=fragment if fragments else None,
                on_progress=progress,
                **keywords,
            )
            packet = take_rollout(
                policy,
                rewards,
                critic,
                terminal=report["status"] == "completed",
                episode_end=True,
            )
            state = {
                "generator": policy.learner.generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "reward": policy.training_components.reward_state,
                "policies": {
                    n: p.get_state()
                    for n, p in policy.policies.items()
                    if n != policy.learner_name
                },
            }
            _send(connection, ("complete", (packet, report, state)))
    except TimeoutError:
        _send(connection, ("timeout", None))
    except BaseException:
        try:
            _send(connection, ("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Samplers:
    def __init__(self, options, seed, count):
        self.workers = []
        context = multiprocessing.get_context("spawn")
        try:
            for index in range(count):
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_worker, args=(child, options, seed, index)
                )
                worker.start()
                child.close()
                self.workers.append((parent, worker))
        except BaseException:
            self.close()
            raise

    def close(self):
        for connection, worker in self.workers:
            # Shutdown also covers a worker blocked at a fragment handshake.
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
            connection.close()
        self.workers.clear()

    def receive(self, index, session):
        from smartsom.experiments.control import boundary

        connection, process = self.workers[index]
        while True:
            boundary()
            if time.time() >= session.deadline:
                raise TimeoutError("training sampling deadline reached")
            if connection.poll(0.1):
                kind, value = _receive(connection)
                if kind == "timeout":
                    raise TimeoutError("training sampling deadline reached")
                if kind == "error":
                    raise RuntimeError(f"sampler {index} failed:\n{value}")
                if kind == "progress":
                    session.progress({**value, "sampling_environment": index})
                    continue
                return kind, value
            if not process.is_alive():
                raise RuntimeError(f"sampler {index} exited: {process.exitcode}")


def step_wave(session):
    from smartsom.config.base_diagnostic import seed_for
    from smartsom.experiments.resource_guard import check
    from smartsom.learning.sampling import learn_packets

    if session.budget_done() or time.time() >= session.deadline:
        return {**session.progress(), "done": True}
    ex, policy = session.ex, session.policy
    count = ex.sampling.num_envs
    # Commit at every requested evaluation/checkpoint boundary, never skip nodes.
    if ex.train_episodes is not None:
        count = min(count, ex.train_episodes - len(session.rows))
    nodes = list(ex.validation.milestones) if ex.validation.unit == "episodes" else []
    nodes += [node.episode for node in ex.evaluation.checkpoints]
    if ex.checkpoint_every_episodes:
        nodes.append(
            (len(session.rows) // ex.checkpoint_every_episodes + 1)
            * ex.checkpoint_every_episodes
        )
    future = [n - len(session.rows) for n in nodes if n > len(session.rows)]
    if future:
        count = min(count, min(future))
    limit = session.rollout_limit()
    if limit is not None:
        count = min(count, limit)
    if not hasattr(session, "samplers"):
        session.samplers = Samplers(
            session.options, session.args.seed, ex.sampling.processes
        )
    weights = policy.learner.sampling_state()
    states = getattr(policy, "sampler_states", {})
    fragments = (
        session.options.learning_policy.learner.params.get("update_trigger", "episode")
        != "episode"
    )
    seen_conditions = {row["condition"] for row in session.rows}
    for index in range(count):
        episode_index = len(session.rows) + index
        condition = (
            session.rng.choice(ex.training_conditions)
            if ex.condition_sampling == "random"
            else ex.training_conditions[episode_index % len(ex.training_conditions)]
        )
        world = seed_for(
            session.args.seed,
            ex.training_world_domain,
            *ex.training_world_prefix,
            episode_index,
        )
        if ex.training_seed_modulus:
            world %= ex.training_seed_modulus
        keywords = {
            "wall_deadline": time.monotonic() + max(0, session.deadline - time.time()),
            "max_ticks": None
            if limit is None
            else limit // count + int(index < limit % count),
            "fragment_ticks": ex.sampling.fragment_ticks,
            "episode_end": ex.episode_end,
            "progress_interval": ex.display.summary_interval,
            "diagnostics": ex.run.diagnostics,
            "metrics_level": ex.run.metrics,
            "metric_interval": ex.run.metric_interval,
            "window_analysis": ex.evaluation.window_analysis,
            "replay_directory": (
                session.root
                / "recordings"
                / "train"
                / f"episode-{episode_index + 1:08d}"
            )
            if ex.replay.train == "all"
            or (ex.replay.train == "first" and condition not in seen_conditions)
            else None,
            "replay_source": session.replay_source,
        }
        seen_conditions.add(condition)
        arguments = (
            *ex.inputs_for(condition, session.factory, session.workload),
            world,
            condition,
            session.drain,
        )
        _send(
            session.samplers.workers[index][0],
            (weights, states.get(index), arguments, keywords, fragments),
        )
    active, completed = list(range(count)), {}
    update_seconds = 0.0
    try:
        while active:
            packets, continuing = [], []
            for index in active:
                kind, value = session.samplers.receive(index, session)
                if kind == "complete":
                    packet, report, state = value
                    completed[index] = report
                    states[index] = state
                    packets.append((packet, True))
                else:
                    packets.append((value, False))
                    continuing.append(index)
            check("learner_update", force=True)
            started = time.monotonic()
            learn_packets(policy, packets)
            update_seconds += time.monotonic() - started
            weights = policy.learner.sampling_state()
            for index in continuing:
                _send(session.samplers.workers[index][0], weights)
            active = continuing
    except TimeoutError:
        session.samplers.close()
        del session.samplers
        session.restore_committed()
        return {**session.progress(), "done": True}
    except BaseException:
        session.samplers.close()
        raise
    policy.sampler_states = states
    for index, result in sorted(completed.items()):
        session.ticks += result["tick"]
        result.update(
            episode=len(session.rows) + 1,
            episodes=len(session.rows) + 1,
            physical_ticks=session.ticks,
            tick_budget=session.budget if ex.train_episodes is None else None,
            episode_budget=ex.train_episodes,
            updates=policy.updates,
            losses=policy.last_losses,
            sampling_environment=index,
            learner_update_seconds=update_seconds / count,
        )
        session.rows.append(result)
    session.commit()
    for result in list(completed.values()):
        session.tracking.log({"phase": "train", **result})
    session.evaluate_periodically()
    progress = session.progress()
    return {**progress, "done": progress["status"] != "training"}
