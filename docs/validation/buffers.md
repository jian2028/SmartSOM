# Week2 item 9 — Finite buffers and blocking acceptance

The implementation adds zero/finite/infinite per-machine pre/post capacity,
exclusive AGV reservations, loaded waiting, automatic unblocking and direct
instantaneous Transfer with AGV off. All movement, processing, event handling
and replay use the existing engine. [ADR 0007](../decisions/0007-finite-buffers-and-blocking.md)
is the full contract, including the deliberate conservative-policy limitations.

This is engineering acceptance of small cases. It does not establish research
performance, scalability or general deadlock avoidance. No dependency was added.

## Independent expected behavior

`data/reference/buffers/hand_expectations.json` fixes the three numeric cases;
their inputs and scripts are separate configuration fixtures. They are checked
against code-built domain inputs and explicit expected event/occupancy times.

| Case / run preset | Fixed processing and movement expectations | Makespan |
|---|---|---:|
| `buffers_direct_zero` | C1/M2 `[0,5)`; A1/M1 `[0,2)` then holds M1 until 5. C exits at 5; A transfers to M2 and B enters M1. A2 and B1 both `[5,6)`; both exit at 6. Pre/post are 0. | 6 |
| `buffers_vehicle_zero` | Both inbound AGVs arrive at 2; V1 unloads A then A processes `[2,6)`. V2 holds B loaded until 6; B processes `[6,7)`. A and B exit at 7 and 8. Pre=0, post=1. | 8 |
| `buffers_post_one` | A processes `[0,2)` into the single post slot. B processes `[2,3)` and holds machine `[3,5)`. Taking A out at 5 automatically moves B to post. C processes `[5,6)`. | 6 |
| Exclusive reservation | B books at 2 while pre=1 is full and arrives at 4. Dispatching A at 2 frees pre; C reserves that slot at 2 and arrives at 6. B cannot borrow it. C unloads first; explicit C dispatch at 6 frees space, then B unloads at 6. | Sequence/ownership checked |
| Real deadlock | One AGV delivers A to a zero-pre/post machine, then carries B toward it. A finishes while B holds the sole vehicle loaded. No physical action/event can free the cycle. | `DeadlockError`, no makespan |

The post=1 script repeats `WaitUntil(5)`: the first call is interrupted by B's
completion at 3; waiting never persists across decisions. Neither hand scripts
nor reference schedules assume persistent waiting.

Additional independent boundaries cover booking-order reversal for simultaneous
arrivals, positive prebuffer refusing machine bypass, same-machine successors,
source pre-slot release only at pickup, and a completed holder booked at 3 with
pickup 5. When another job leaves post at 3, that holder stays bound on its
machine until 5, even though the machine goes down at 4. Its processing already
ended at 3; no spurious pause/resume is recorded.

A two-machine zero-buffer cycle also distinguishes **policy failure** from
physical deadlock: conservative SPT has no safe booking and no future event,
although two legal early vehicle trips can free the positions and finish at 3.
The policy failure is explicit; the engine does not claim these legal trips are
impossible or perform them automatically. Scripts retain them.

## Replay, combination and compatibility checks

- Direct, policy and action replay agree exactly. V2 schedule replay checks
  processing, every vehicle's pickup/arrival/unload, direct transfers and action
  order. Reordering containers preserves semantic sequences and results.
- Changed same-tick action order, missing/duplicate actions, wrong sequences or
  scalar types, missing trips, wrong travel/arrival/unload/source and processing
  inconsistencies fail. Dynamic occupancy is checked through `step()`, followed
  by exact final record comparison; no recovery or implicit missing move occurs.
- Both movement modes cover all 8 JA/MB/UPT combinations. JA uses both existing
  decision triggers; both automatic providers run through configured evidence.
  Multi-mode dispatch/rerouting, machine downtime, completed actual work and
  immutable snapshots are checked. Future repair/unloading remains hidden.
- Strict configuration rejects invalid/unknown capacities and references before
  Simulator/run-directory allocation. `validate` creates no run. Script/CP
  compatibility, export/reimport, CLI codes, provider/replay/writer failure
  evidence and output progress counts are checked.
- Changing working directory, `PYTHONHASHSEED`, machine/buffer/matrix/job/mode
  input order preserves domain values, all seeds, observations and execution.
- `disabled_golden.json` was obtained by executing an unmodified `git archive`
  of pre-item-9 main `df3d0cfb542757204069a718ef816efe04657b84` in the locked
  environment. Ten old scenarios retain factory/workload/seed and complete
  schedule/action/trace/makespan digests, including v1 execution serialization.
  Explicit all-infinite AGV is checked against those same results. Existing
  pre-item-8 goldens and all prior core tests remain active.

## Actual external comparisons and limits

`data/reference/buffers/external.json` freezes the actual matched results,
reference package/source hashes, fixtures and differences. Reproduction uses
`scripts/validation/buffer_references.py` against unchanged external source.

**PyJobShop 0.0.9 / OR-Tools 9.12.4544:** the
[official blocking example](https://pyjobshop.org/stable/examples/permutation_flow_shop.html#blocking)
uses idle-extended tasks linked to successor starts. Our independent fixed-start
microcase applies that method and actually solves `OPTIMAL / objective 6 / bound 6`
with one worker and seed 0. A1's returned resource interval is `[0,5)`, containing
2 processing ticks and 3 idle ticks. SmartSOM records net completion at 2 and
resource release at Transfer tick 5. These are deliberately separate quantities.
The other three intervals agree with the hand table above. The required CP CI
suite executes this solve and comparison; it is not replaced by a skipped test.

**JobShopLab commit `764af47cb5ca3ab7666d08ac8b84385207bfffd9`:** the probe imports
and executes the actual buffer helpers and completion handler from the
[pinned source](https://github.com/proto-lab-ro/jobshoplab/tree/764af47cb5ca3ab7666d08ac8b84385207bfffd9).
It verifies capacity and selecting the second FLEX-buffer job. Completion with a
free post slot moves the job and releases the machine. Full and zero postbuffers
instead raise `BufferFullError`; raw error messages and input state are preserved.
These results **do not validate full blocking, reservations or loaded waiting**.
Those contracts use the independently specified hand cases and engine invariants.
The namespace-only bootstrap avoids Gym/render setup; NumPy is pinned at 2.5.3 via
the existing isolated validation requirements. External source is not modified
and JobShopLab is not a project/runtime dependency. No fully matching external
simulator was established by the bounded search.

## Fresh checks before commit

- Focused buffer, existing transport and CP checks passed; the final complete
  suite also includes the added source-holding and policy-exhaustion cases.
- Base environment: **649 passed, 12 skipped**. The skips are the optional CP
  suite in the environment without PyJobShop/OR-Tools.
- Locked CP environment with `SMARTSOM_REQUIRE_CP=1`: **661 passed**, no skips.
  Actual adapter regressions retain ft06 **55/55/55**, official FJSP **6/6/6**,
  Mk01 **40/40/40**, plus the new blocking microcase **6/6/6**.
- Ruff, formatting, `uv lock --check`, both environments'
  dependency checks and `git diff --check` pass.
- The bounded configured check completes **51 runs**: three hand scripts plus
  two movement modes × 12 module/trigger combinations × two baselines. Every run
  also compares direct policy execution, action replay and full schedule replay.
  Development evidence is under `artifacts/item9/precommit-configured/`; it records
  the parent commit and dirty implementation honestly, rather than treating that
  parent commit as the new feature's source.

## Reproduction and final source identity

```bash
uv sync --locked
uv run pytest -q tests/unit/test_buffer_inputs.py tests/unit/test_buffers.py tests/unit/test_buffer_lifecycle.py
uv run python scripts/validate_buffers.py --output-dir artifacts/item9/configured-NEW
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv lock --check
uv pip check
uv sync --locked --extra cp
SMARTSOM_REQUIRE_CP=1 uv run --no-sync pytest -q
uv run --no-sync python scripts/validation/buffer_references.py \
  --source /absolute/path/to/pinned/jobshoplab \
  --output artifacts/item9/external-NEW.json
git diff --check
```

The reference script requires the existing pinned validation NumPy version;
regular base usage does not. Output directories/files must be new. There is no
batch API hidden in these bounded acceptance scripts.

The item-9 commit includes code, tests, these inputs, ADR, examples and this
record. After that commit, rerun the 51 configured checks, focused hand/lifecycle
checks, actual static CP/reference acceptance and the external probe; save their
exit statuses and new commit identity under `artifacts/item9/<new-commit>/`.
`artifacts/item9/final-checks.json` is the local completion index after those
checks finish. Generated runtime directories are not committed. The final commit
identity is intentionally recorded there rather than embedded self-referentially
in this document. Original governance and paper edits remain outside the commit.
