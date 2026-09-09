"""Ordered simulation streams, optionally executed by owned spawn processes.

Learners, RNG for policy proposals, and evidence writers stay in the coordinator.
Workers only construct and advance the existing environment protocols.
"""

import multiprocessing
import signal
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

from smartsom.config.codec import digest


@dataclass(frozen=True, slots=True)
class EpisodeSnapshot:
    episode_index: int
    stream_id: int
    local_episode: int
    input: object
    steps: tuple
    trace: tuple | None
    bindings: tuple
    total_reward: float
    reason: str | None
    result: object
    finished: bool
    possible_agents: tuple[str, ...]
    extension_state: dict | None = None

    @property
    def simulator(self):
        return SimpleNamespace(trace=self.trace) if self.trace is not None else None

    @property
    def projection(self):
        return SimpleNamespace(bindings=self.bindings)

    @staticmethod
    def policy_for_agent(agent):
        return "machine_policy" if agent.startswith("machine:") else "agv_policy"


class SamplingWorkerError(RuntimeError):
    """Original worker exception identity/traceback retained across the process edge."""

    def __init__(self, error):
        self.original_type = error["type"]
        self.original_traceback = error["traceback"]
        self.snapshot = error["snapshot"]
        super().__init__(f"{error['type']}: {error['message']}\n{error['traceback']}")


class _Stream:
    def __init__(self, resolved, stream_id, num_envs):
        self.resolved, self.stream_id, self.num_envs = resolved, stream_id, num_envs
        self.local_episode = -1
        self.resource = resolved.algorithm.algorithm.provider == "rllib.resource_ppo"
        if self.resource:
            from smartsom.learning.pettingzoo import SmartSOMParallelEnv

            kind, kwargs = SmartSOMParallelEnv, {}
        else:
            from smartsom.learning.gymnasium import SchedulingEnv

            kind, kwargs = (
                SchedulingEnv,
                {
                    "strict_actions": True,
                    "observation_kind": "masked"
                    if resolved.algorithm.algorithm.provider == "rllib.ppo"
                    else "plain",
                },
            )
        from smartsom.learning.training_extensions import environment_arguments

        self.env = kind(
            resolved.episode(stream_id).input,
            resolved.algorithm.algorithm.projection,
            limits=resolved.run.budget.limits(),
            episode_source=lambda index: resolved.episode(index).input,
            **kwargs,
            **environment_arguments(resolved),
        )

    def reset(self):
        self.local_episode += 1
        # Stream identity is independent of reset/completion order. Stream zero in
        # a single-env run retains the original training-episode seed recipe.
        self.env.episode_index = self.local_episode * self.num_envs + self.stream_id - 1
        return self.env.reset()

    def snapshot(self):
        env = self.env
        return EpisodeSnapshot(
            env.episode_index,
            self.stream_id,
            self.local_episode,
            getattr(env, "input", env.base_input),
            tuple(env.steps),
            tuple(env.simulator.trace) if env.simulator is not None else None,
            tuple(env.projection.bindings),
            getattr(env, "total_reward", 0.0),
            getattr(env, "reason", None),
            env.result,
            env.finished,
            tuple(env.possible_agents) if self.resource else (),
            env.extension_state_dict(),
        )

    def state(self):
        snap = self.snapshot()
        return {
            "stream_id": self.stream_id,
            "local_episode": self.local_episode,
            "indices": [
                s.indices if self.resource else s.action_index for s in self.env.steps
            ],
            "snapshot_sha256": digest(snap),
            **(
                {"extension_state": snap.extension_state}
                if snap.extension_state
                else {}
            ),
        }

    def restore(self, state):
        if state["stream_id"] != self.stream_id:
            raise ValueError("sampling stream identity differs on restore")
        from smartsom.learning.training_extensions import (
            restore_initial_state,
            verify_restored_state,
        )

        extension = state.get("extension_state")
        restore_initial_state(self.env, extension)
        self.local_episode = state["local_episode"] - 1
        self.reset()
        for action in state["indices"]:
            self.env.step(dict(action) if self.resource else action)
        verify_restored_state(self.env, extension)
        if self.state() != state:
            raise ValueError("sampling stream replay differs from saved activity")
        return self.snapshot()

    def command(self, command, value):
        if command == "reset":
            return self.reset(), self.snapshot()
        if command == "step":
            return self.env.step(value), self.snapshot()
        if command == "state":
            return self.state()
        if command == "restore":
            return self.restore(value)
        if command == "spaces":
            if self.resource:
                return (
                    self.env.observation_spaces,
                    self.env.action_spaces,
                    self.env.possible_agents,
                )
            return self.env.observation_space, self.env.action_space, ()
        if command == "mask":
            return self.env.action_masks()
        raise ValueError(f"unknown sampling operation {command}")


def _worker(connection, resolved, indices, num_envs, registrations):
    from smartsom.learning.extensions import install_registrations

    install_registrations(registrations)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    streams = {index: _Stream(resolved, index, num_envs) for index in indices}
    try:
        while True:
            requests = connection.recv()
            if requests is None:
                return
            results = []
            for index, command, value in requests:
                try:
                    results.append(
                        (index, True, streams[index].command(command, value))
                    )
                except BaseException as exc:
                    results.append(
                        (
                            index,
                            False,
                            {
                                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                                "snapshot": streams[index].snapshot(),
                            },
                        )
                    )
            connection.send(results)
    finally:
        for stream in streams.values():
            stream.env.close()
        connection.close()


class OrderedSamplingPool:
    def __init__(self, resolved, num_envs, processes, evidence):
        from smartsom.learning.training_extensions import learner_scale

        self.num_envs, self.processes, self.evidence = num_envs, processes, evidence
        self.learner_scale = learner_scale(resolved)
        self.snapshots = [None] * num_envs
        self.connections, self.workers = [], []
        self.local = {}
        if processes:
            from smartsom.learning.extensions import export_registrations

            spec = resolved.algorithm.algorithm
            registrations = export_registrations(spec.extensions, spec.provider)
            context = multiprocessing.get_context("spawn")
            for worker_id in range(processes):
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_worker,
                    args=(
                        child,
                        resolved,
                        list(range(worker_id, num_envs, processes)),
                        num_envs,
                        registrations,
                    ),
                    daemon=True,
                )
                worker.start()
                child.close()
                self.connections.append(parent)
                self.workers.append(worker)
        else:
            self.local = {
                index: _Stream(resolved, index, num_envs) for index in range(num_envs)
            }
        try:
            self.spaces = self.call(
                "spaces", {index: None for index in range(num_envs)}
            )
        except BaseException:
            self.close()
            raise

    def call(self, command, values):
        if self.processes:
            pending = []
            for worker_id, connection in enumerate(self.connections):
                requests = [
                    (index, command, values[index])
                    for index in sorted(values)
                    if index % self.processes == worker_id
                ]
                if requests:
                    connection.send(requests)
                    pending.append(connection)
            responses = []
            for connection in pending:
                responses.extend(connection.recv())
            result, errors = {}, []
            for index, success, value in sorted(responses):
                if not success:
                    self.snapshots[index] = value["snapshot"]
                    errors.append(value)
                else:
                    result[index] = value
                    self._observe(command, index, value)
            self._publish()
            if errors:
                raise SamplingWorkerError(errors[0])
            return result
        result = {}
        try:
            for index in sorted(values):
                try:
                    value = self.local[index].command(command, values[index])
                except BaseException:
                    self.snapshots[index] = self.local[index].snapshot()
                    raise
                result[index] = value
                self._observe(command, index, value)
            return result
        finally:
            self._publish()

    def _observe(self, command, index, value):
        if command in ("step", "reset"):
            self.snapshots[index] = value[1]
        elif command == "restore":
            self.snapshots[index] = value

    def _publish(self):
        self.evidence.active_envs = self.snapshots.copy()
        self.evidence.active_env = self.snapshots[0]

    def reset(self, indices=None):
        result = self.call(
            "reset",
            {
                index: None
                for index in range(self.num_envs)
                if indices is None or index in indices
            },
        )
        for index, (_, snapshot) in result.items():
            self.snapshots[index] = snapshot
        self.evidence.active_envs = self.snapshots.copy()
        self.evidence.active_env = self.snapshots[0]
        return {index: value[0] for index, value in result.items()}

    def step(self, actions):
        result = self.call("step", dict(enumerate(actions)))
        for index, (_, snapshot) in result.items():
            self.snapshots[index] = snapshot
            if snapshot.finished:
                self.evidence.episode(snapshot)
        self.evidence.active_envs = self.snapshots.copy()
        self.evidence.active_env = self.snapshots[0]
        return [result[index][0] for index in range(self.num_envs)]

    def state(self):
        return list(
            self.call("state", {index: None for index in range(self.num_envs)}).values()
        )

    def restore(self, states):
        self.snapshots = list(self.call("restore", dict(enumerate(states))).values())
        self.evidence.active_envs = self.snapshots.copy()
        self.evidence.active_env = self.snapshots[0]

    def close(self):
        for stream in self.local.values():
            stream.env.close()
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
