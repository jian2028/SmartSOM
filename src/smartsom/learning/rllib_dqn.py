"""Ray 2.60 DQN with replay state in the same official checkpoint.

State export/import and empty fragment-cache admission are extended. Training,
targets, exploration, replay sampling and checkpoint storage remain RLlib's.
Import this optional adapter only from RLlib execution paths.
This is same-version persistence, not a legacy-checkpoint compatibility layer.
"""

from ray.rllib.algorithms.dqn import DQN
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.execution.segment_tree import MinSegmentTree
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_episode_buffer import (
    MultiAgentPrioritizedEpisodeReplayBuffer,
)

REPLAY_COMPONENT = "replay_buffer"
REPLAY_REQUIRED = "smartsom_replay_required"


class RestorableMultiAgentPrioritizedEpisodeReplayBuffer(
    MultiAgentPrioritizedEpisodeReplayBuffer
):
    """Restore the inverse index still omitted by Ray 2.60's state loader."""

    def add(self, episodes, weight=None):
        if isinstance(episodes, MultiAgentEpisode):
            episodes = [episodes]
        normalized = []
        for episode in episodes:
            state = episode.get_state()
            rewards = state["_hanging_rewards_end"]
            actions = state["_hanging_actions_end"]
            empty = [
                agent
                for agent, reward in rewards.items()
                if agent not in actions and reward == 0.0
            ]
            if empty:
                # Ray 2.60 cut() can leave a zero reward without a pending action.
                # concat_episode() mistakes that key for an unfinished transition.
                # Normalize only empty cache entries, never trajectories or rewards.
                rewards = rewards.copy()
                for agent in empty:
                    del rewards[agent]
                state["_hanging_rewards_end"] = rewards
                episode = MultiAgentEpisode.from_state(state)
            normalized.append(episode)
        return super().add(normalized, weight=weight)

    def set_state(self, state):
        super().set_state(state)
        # The official setter restores the forward map but omits its inverse,
        # which add() needs when evicting checkpointed episodes. This internal
        # attribute dependency is explicit and must be rechecked on Ray upgrades.
        self._sample_idx_to_tree_idx = {
            sample_index: tree_index
            for tree_index, sample_index in self._tree_idx_to_sample_idx.items()
        }


def _includes_replay(components, not_components):
    def names(value):
        return [value] if isinstance(value, str) else list(value or [])

    included, excluded = names(components), names(not_components)
    if any(name.startswith(REPLAY_COMPONENT + "/") for name in included + excluded):
        raise ValueError("replay_buffer is an indivisible checkpoint component")
    return (components is None or REPLAY_COMPONENT in included) and (
        REPLAY_COMPONENT not in excluded
    )


class ReplayCheckpointDQN(DQN):
    """Add replay to root state without replacing RLlib's checkpoint writer."""

    def _complete_replay_state(self, state):
        # Ray 2.60 restores this MARL counter as defaultdict(list), although its
        # initial default is int. Materialize the implicit zeros for the frozen
        # module inventory, so warmup/empty checkpoints use only existing keys.
        key = "sampled_timesteps_per_module"
        if key not in state:
            return state
        counts = dict(state[key])
        for module_id in self.config.policies:
            counts.setdefault(module_id, 0)
        state = {**state, key: list(counts.items())}
        # Ray 2.60 restores the missing-module minimum-tree factory as a SUM
        # tree. Materialize the official neutral minimum trees for our frozen
        # module inventory; preserve every existing tree and priority unchanged.
        trees_key = "_module_to_min_segment"
        if trees_key in state:
            trees = dict(state[trees_key])
            # Official root tree serialization stores both leaves and parents.
            capacity = len(state["_min_segment"]) // 2
            for module_id in self.config.policies:
                if module_id not in trees:
                    trees[module_id] = MinSegmentTree(capacity)
            state = {**state, trees_key: list(trees.items())}
        return state

    def get_state(self, components=None, *, not_components=None, **kwargs):
        include_replay = _includes_replay(components, not_components)
        if include_replay and not self.config.store_buffer_in_checkpoints:
            raise ValueError("SmartSOM DQN requires store_buffer_in_checkpoints=True")
        state = super().get_state(
            components=components, not_components=not_components, **kwargs
        )
        state[REPLAY_REQUIRED] = include_replay
        if include_replay:
            state[REPLAY_COMPONENT] = self._complete_replay_state(
                self.local_replay_buffer.get_state()
            )
        return state

    def set_state(self, state):
        required = state.get(REPLAY_REQUIRED, "training_iteration" in state)
        if type(required) is not bool:
            raise ValueError("invalid replay checkpoint declaration")
        if required and REPLAY_COMPONENT not in state:
            raise ValueError("complete DQN recovery requires checkpoint replay state")
        replay = None
        if REPLAY_COMPONENT in state:
            if not self.config.store_buffer_in_checkpoints:
                raise ValueError(
                    "cannot restore replay with checkpoint storage disabled"
                )
            # Official episode buffers populate episodes only when initially empty.
            # Construct a fresh official buffer so restoring into a trained Algorithm
            # replaces its contents instead of mixing old episodes with new indices.
            replay = from_config(ReplayBuffer, self.config.replay_buffer_config)
            replay.set_state(self._complete_replay_state(state[REPLAY_COMPONENT]))
        super().set_state(state)
        if replay is not None:
            self.local_replay_buffer = replay
