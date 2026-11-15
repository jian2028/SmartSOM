"""Factory progress on public RLlib callbacks and checkpointed MetricsLogger."""

import time

from ray.rllib.callbacks.callbacks import RLlibCallback


def factory_infos(episode, index):
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode

    infos = episode.get_infos(index)
    return (
        (infos,) if isinstance(episode, SingleAgentEpisode) else tuple(infos.values())
    )


def _tick(episode, index):
    return max(info["physical_tick"] for info in factory_infos(episode, index))


class FactoryProgress(RLlibCallback):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        algorithm.metrics.log_value(
            # Logged once per Algorithm construction; retain it across normal
            # metric reductions, including a first report without evaluation.
            "smartsom_algorithm_initialized_unix",
            time.time(),
            reduce="lifetime_sum",
        )

    def on_episode_start(self, *, episode, env_runner, metrics_logger, **kwargs):
        if not env_runner.config.in_evaluation:
            metrics_logger.log_value(
                "smartsom_physical_ticks", _tick(episode, -1), reduce="sum"
            )

    def on_episode_step(self, *, episode, env_runner, metrics_logger, **kwargs):
        if not env_runner.config.in_evaluation:
            metrics_logger.log_value(
                "smartsom_physical_ticks",
                _tick(episode, -1) - _tick(episode, -2),
                reduce="sum",
            )

    def on_episode_end(self, *, episode, env_runner, metrics_logger, **kwargs):
        if not env_runner.config.in_evaluation:
            metrics_logger.log_value("smartsom_episodes", 1, reduce="sum")

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        # The Algorithm's official checkpoint contains this MetricsLogger.
        # Counters survive restore without serializing simulator state or a
        # second application training-state store.
        sampled = result.get("env_runners", {})
        result["algorithm_initialized_unix"] = metrics_logger.peek(
            "smartsom_algorithm_initialized_unix", default=0.0
        )
        for name in ("physical_ticks", "episodes"):
            key = "smartsom_" + name + "_lifetime"
            metrics_logger.log_value(
                key, sampled.get("smartsom_" + name, 0), reduce="lifetime_sum"
            )
            result[name] = metrics_logger.peek(key)
