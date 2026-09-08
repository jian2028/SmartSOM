# Resource-agent MARL acceptance

The first checkpoint adds public resource projection, deterministic proposal
coordination and the optional PettingZoo Parallel API. Training/checkpoint/run/study
integration is the second checkpoint; interface tests alone do not complete item 13.
See [ADR 0012](../decisions/0012-resource-agents-and-joint-proposals.md).

## IDETC field comparison

The frozen runtime is the **inner** `jobshoplab/jobshoplab/env/env_wrapper.py` at
729bc692c85781be275f42144ce582449b875d98. Its package shim prioritizes that directory.
Both verified byte hashes are in the committed source manifest. No external code
is needed for tests or execution.

| Frozen field/group | Current interpretation/difference |
| --- | --- |
| job_running, job_progression | Public processing/paused flag and completed operation count |
| job_executed_on_machine | Binary history reconstructed from completed semantic modes |
| machine_progression | Distinct completed jobs per machine, not operation count |
| available_jobs | Released, unfinished, not processing/paused; not a legality mask |
| current/global time | One public clock; remove duplicate feature |
| global input/output fill | Counts with configured scale; these areas are unlimited |
| local machine/buffer state | Current up/down, holding phase, membership and reservations |
| machine_remaining_time | Chosen nominal duration and elapsed wall simulation time; no internal expected completion |
| AGV remaining time | Known pickup/arrival only; unload waiting remains unknown |
| candidate_job/remaining_ops/duration | Per-resource full semantic choices, with explicit mode/destination and nominal/travel parameters |
| current_transition | Replaced by explicit information for every candidate |
| full hidden job directory/normalizers | Fixed declared capacities, reveal-bound slots and configured scales |
| truncated lists/modulo/fallback actions | Rejected; no truncation, remapping or conflict resampling |

## Independent checks

Crossing jobs start C1/M1 [0,2), D1/M2 [0,3) in the same joint round. The next
round starts C2/M2 [3,4), D2/M1 [3,5), makespan **5**. Two machines claiming A1
accept M1's action and reject M2 without choosing B1 as a fallback. Two zero-time
AGV proposals for A accept only the first, even though the second would become
a legal reroute after delivery. Processing beats a conflicting transport proposal.

All legal JA/MB/UPT/AGV/buffers/quality/holding switch combinations are checked,
including both arrival triggers and direct Transfer ownership. Fixed semantic
choices compare exact core trace, action replay and schedule/quality replay.
NOOP, public waiting, stalls, atomic rejection, reveal/hidden truth, fixed capacity,
initial advancement, limits and complete lifecycle are independently checked.

PettingZoo 1.27's official Parallel API test runs directly on the environment.
Its seed helper ignores masks, so a **test-only sampler** supplies the public mask
before drawing an action; env.step remains strict. That helper also tests only
one transition due to any(dict) on terminal dictionary keys. Independent seeded
full-episode comparisons supplement it; no external test source is modified.

```sh
uv sync --locked --extra pettingzoo
SMARTSOM_REQUIRE_PETTINGZOO=1 uv run --no-sync pytest -q \
  tests/unit/test_resource_projection.py tests/unit/test_pettingzoo.py
```

The real PPO gate is fixed at seed101, 4096 joint environment steps, two 64/64
role networks and the existing 4-job/9-operation micro. Evaluation uses study
seed202, replicas0–4, MARL+SPT (10 runs). No superiority requirement or claim of
equal sampling budgets with centralized PPO. Fixed-budget failure remains failure.
Actual Linux execution remains pending until run, regardless of CI configuration.

First-checkpoint verification on macOS: 305 focused resource/Parallel checks;
1488 full locked learning/CP tests; 1169 base tests and 19 explicit optional skips.
Ruff, formatting, dependency consistency, lock check and diff checks passed.
The first sandboxed full attempt failed because Ray could not bind localhost and
was interrupted; the completed unsandboxed local rerun is the passing evidence.
These are engineering checks, not the new MARL training or integrated acceptance.
