# Week2 item 8 — Fixed-matrix AGV acceptance

Scope: multiple capacity-one AGVs, directed fixed integer travel, unlimited
waiting areas, explicit queue rerouting, arrivals/outages/processing uncertainty,
and exact action/full-execution replay. The contract is
[ADR 0006](../decisions/0006-fixed-matrix-transport-and-execution-replay.md).
Finite buffers and blocking are item 9; transport CP remains unsupported.

## Hand reference and configured examples

The independent `data/reference/transport/hand_schedule.json` specifies one job
with M1/2 then M2/3 and V1 initially at M2:

| Stage | Expected ticks |
| --- | --- |
| Empty M2 → input | 0–2 |
| Loaded input → M1 | 2–5 |
| Processing A on M1 | 5–7 |
| Loaded M1 → M2, pickup at 7 | 7–11 |
| Processing B on M2 | 11–14 |
| Loaded M2 → output | 14–15 |

Makespan is **15**; processing finishes at **14**. The hand table was constructed
independently of engine output. Factory, workload and table are versioned together.

```bash
uv run smartsom validate configs/runs/transport_hand.yaml
uv run smartsom run configs/runs/transport_hand.yaml       # 15
uv run smartsom run configs/runs/transport_reroute.yaml    # 7
uv run smartsom run configs/runs/transport_combined.yaml
uv run python scripts/validate_transport.py --output-dir artifacts/item8/acceptance
```

The output directory argument must be new. The bounded acceptance script runs
26 configured attempts: hand and down-machine rerouting, then both SPT and
first-feasible for all 8 JA/MB/UPT combinations. JA-enabled cells exercise both
decision triggers. Each attempt compares code execution, action replay, exported
schedule replay and reordered schedule records, and saves ordinary run evidence.
This is an acceptance script, not a batch API or scale claim.

The rerouting fixture first delivers to down M1, moves its pending job to up M2,
selects M2's mode at dispatch, and delivers to output at **7**. Unit checks add busy
queues, non-FIFO processing, multiple vehicles, same-machine successors, slow-mode
selection and repeated zero-tick reroutes before the final dispatch. Invariants
check one job position, binding, active vehicle phase/event, physical continuity,
matrix duration and final output coverage after each transition.

## Fixed external evidence

`data/reference/transport/jobshoplab.json` saves the independent instance, initial
state, chosen transitions, raw sub-states/final states, loaded source file SHA-256
values and NumPy version for JobShopLab commit
`764af47cb5ca3ab7666d08ac8b84385207bfffd9`. The repository URL is recorded in that
fixture. Validation imports its actual state-machine code unchanged; namespace
bootstrap bypasses only the optional umbrella Gym/render imports. No JobShopLab
dependency enters SmartSOM or its base environment.

```bash
# In the locked CP environment, or an isolated Python 3.12 environment with
# scripts/validation/jobshoplab-transport-requirements.txt (NumPy 2.5.3):
uv run --no-sync python scripts/validation/jobshoplab_transport.py \
  --source /path/to/pinned/jobshoplab \
  --output artifacts/item8/jobshoplab.json
```

Observed external transport phases are **(0,2,5), (7,7,11), (14,14,15)**, matching
the hand intervals and SmartSOM. The external final job is at output, but its final
clock resets to **14** after a delivery sub-state at **15**. Its source labels
sub-states diagnostic and potentially semantically incomplete. We retain all raw
records, verify the actual phase times and processing intervals, and do not claim
whole-Gym, termination-clock or action-trajectory equivalence. The existing
JobShopLab tests are not the numeric oracle. External source remains unmodified.

## Compatibility and failure checks

`disabled_golden.json` was obtained from integrated pre-item8 source commit
`28f53e83a4c60b5366bcbac0094e110afef547f6`, exported read-only to a temporary
directory. Twelve JSP/FJSP/arrival/UPT/MB cases match exactly in factory/workload
digests, all named seeds, full schedule/actions/trace and makespan. These fixed
goldens remain a regression test. Existing seed-generation goldens are unchanged.

New tests additionally cover strict matrix/schema validation before execution,
atomic invalid transports/dispatches, unbound versus bound queues, information
hiding, snapshot immutability, same-tick event order, input permutation, working
directory/hash seed changes, script failures, replay mismatch and writer failure.
Transport off retains the old full trace; zero-matrix transport on retains its own
transport records and only promises matched processing schedule/makespan.

Fresh pre-commit verification: **62 new transport/input tests**, **582 base tests
passed** (11 optional CP tests skipped only there), and **593 tests passed** with
`SMARTSOM_REQUIRE_CP=1` in the locked CP environment. Actual CP objective/bound/
replay remain ft06=55, official FJSP=6, Mk01=40. The 26 configured acceptance runs
and direct pinned JobShopLab probe passed. Ruff, formatting, `uv lock --check`,
both environments’ `uv pip check`, and `git diff --check` passed.

The quality gates are focused tests, base-environment full pytest, required CP
full pytest (ft06 **55**, official FJSP **6**, Mk01 **40**), Ruff, formatting, locked
dependency checks and `git diff --check`. Generated reports remain under ignored
`artifacts/item8/`; post-commit runs must record the new source commit. Dirty
governance/paper files are reported as dirty rather than hidden or included in this
feature commit. Passing micro cases establish these engineering contracts only.
