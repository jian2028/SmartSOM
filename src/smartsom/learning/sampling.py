"""Detached trajectory packets; physical and parameter ownership stay separate."""

import copy

from smartsom.learning.subjects import subject_key


def take_rollout(policy, rewards, critic, *, terminal, episode_end):
    learner = policy.learner
    records = list(learner.records)
    timeline = copy.copy(rewards)
    timeline.final_requests = dict(rewards.final_requests)
    retained = []
    if getattr(learner, "needs_next_request", False) and not episode_end:
        # A fragment is not truncation. Keep unresolved actions until the SAME
        # subject's next decision; never manufacture a candidate at a cut.
        tails = {}
        for index, row in enumerate(records):
            tails[
                subject_key(row["request"], policy.options.learning_policy.split_stages)
            ] = index
        indices = set(tails.values())
        retained = [r for i, r in enumerate(records) if i in indices]
        records = [r for i, r in enumerate(records) if i not in indices]
        timeline.final_requests = {
            key: learner.records[index]["request"] for key, index in tails.items()
        }
    learner.discard_rollout()
    for row in retained:
        learner.records.append(row)
    return records, timeline, terminal, critic


def learn_packets(policy, packets):
    learner = policy.learner
    if not getattr(learner, "needs_next_request", False):
        return learner.update_rollouts([packet[0] for packet in packets])
    retained = learner.records
    for (records, rewards, terminal, critic), episode_end in packets:
        learner.records = records
        learner.update(
            rewards, terminal=terminal, final_critic=critic, episode_end=episode_end
        )
    learner.records = retained
    return learner.last_losses


def fragment_callback(policy, rewards, critic):
    packet = take_rollout(policy, rewards, critic, terminal=False, episode_end=False)
    learn_packets(policy, [(packet, False)])
