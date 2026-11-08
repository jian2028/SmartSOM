"""Named input layout shared by observation construction and the network."""

SCHEMA = "example.local-statistics/v1"
BASE_FIELDS = ("local_jobs", "local_overdue_jobs")
CANDIDATE_FIELDS = ("related_jobs", "mean_due_slack")
SOCIAL_FIELDS = ("related_peer_busy_fraction", "related_peer_pre_jobs")


def layout(social):
    return {
        "base": BASE_FIELDS,
        "candidates": CANDIDATE_FIELDS,
        "social": SOCIAL_FIELDS if social else (),
    }


def inventory(buffer):
    return tuple(job for rows in buffer["inventory"].values() for job in rows)


def related_jobs(decision, action, jobs, vehicles, buffers):
    if action in (("STOP",), ("EMPTY_WAIT",)):
        return ()
    if decision.group == "machine" or (
        decision.group in ("post", "input") and decision.stage == "service"
    ):
        return (action[0],)
    if decision.group == "dispatcher":
        # Loaded vehicles are evaluating destinations for their own carried job.
        carried = vehicles[decision.owner]["job"]
        return (carried,) if carried else tuple(buffers[action[0]]["ready"])
    if decision.group == "pre":
        carried = vehicles[action[0]]["job"]
        return (carried,) if carried else ()
    return tuple(buffers[decision.owner]["ready"])
