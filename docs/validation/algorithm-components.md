# Algorithm components: engineering acceptance

This record covers the component refactor after `40ea4f009f43d47251049213df3d325bb5c89e89`, including the separately preserved Small/Medium map edits. It does not report a formal research experiment or algorithm superiority. Hosted Linux/macOS acceptance is reported by the final PR checks, separately from local results.

## Source and review

The pre-change source archive includes the dirty map changes, examples, tests, `pyproject.toml` and `uv.lock`. Its SHA256 is `6d81cb0b4117c68be74a2fee818ee45a44ed2cfbc6a6e0082d9c52f436324e17`; see the [archive index](../archive/index.json).

`research-code-review` used independent engineering, scientific-semantics and validation reviewers on complete source snapshots including untracked files. The second snapshot manifest SHA256 was `af22a09998e1af34bab4426c8fa8994eb212299817ec7fd919ca9d2166f8d1ba`. The final migration increment was independently rechecked in snapshot `de9fb5d9910085f8b13a77f0a95d4dd60957d088b7bec6d9b34801096617a204`.

Confirmed findings were fixed and independently closed:

- Different entry files in the same source bundle now have different checkpoint identities; relocation of the same relative entry remains valid.
- Migration validates legacy raw-file SHA256 before importing source, and only declares strategies actually selected by the old bindings.
- Centralized migration retains shared quality penalties/scales, while rejecting distributed action declarations.
- Matrix-only external networks are validated, copied and relocated together with base components. Coupled old STOP axes retain two variants rather than becoming independent Pre/Post axes.
- Migration preserves `~` and original-directory relative paths, including calibration/checkpoint matrix axes.

Non-implementer scenarios independently checked six/eight migrated matrix members after original sources were deleted, nested packages/resources after relocation, wrong-source rejection, checkpoint identity and policy state, per-policy projection without changing candidates, invalid factory signatures, and 70 physical reward ticks with only eight learning decisions among 262 mixed decisions. No unresolved P0/P1/P2 finding remained in these reviews.

## Local verification

Use the pinned Python 3.12 locked environment. The engineering commands are:

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
WANDB_MODE=disabled uv run --no-sync python -m pytest -q
WANDB_MODE=disabled uv run --no-sync python -m pytest tests/unit/test_algorithm_components.py -q
```

Tracking-specific tests explicitly select offline/disabled behavior or inject controlled clients. These checks do not establish real cloud synchronization. Public lifecycle tests use real processes and require host process inspection.

The full suite exercises Base/SL × PPO/DQN short training, evaluation and safe restoration; DQN replay/target state; centralized reservations, cancellation and redirection; launch/profile/diff/completion/catalog; native/Ray lifecycle; SDK events and RNG isolation. The component-specific regressions additionally cover migration, source bundles, action ablations, mixed policy projections and a custom learner through the shared runtime.

Runtime snapshot full suite: **1,572 passed, 4 platform skips in 408.77 s** on macOS/Python 3.12.13. The later migration-only fix was independently reviewed and its final **15 component regressions passed in 3.09 s** in an existing lean environment without Torch, Ray, Gymnasium, W&B, TensorBoard or Qt. Ruff lint/format and whitespace checks passed. The only post-review test adjustment used a metadata-only network fixture to keep that test in the fast owner; full tensor-network behavior remains in integration acceptance.

All fast-owner selectors also ran in that lean environment on macOS: **971 passed, 2 platform skips in 100.59 s**. The platform-enforcing CI runner correctly refused to label this host as Ubuntu; the local selector run is not a substitute for hosted Linux CI.

Final Python source inventory SHA256: `ea7bd99189b53afdc981bb3227af88bfccf38351826f802c937591259abb5196` (sorted path → file SHA256 JSON). The map-only baseline tests separately passed for Small/Medium.

## Differential and resource evidence

A matched probe uses Small, eight jobs, world seed 19, learner seed 7, 160 physical ticks, one update epoch and empty WAIT enabled. For each of PPO Base, PPO SL, DQN Base and DQN SL, the pre-change and final runtime produce identical action identities, reward sequences, update counts and updated model tensor bytes. Separate fixed-world tests preserve full trace hashes, including resource ownership and time, against the earlier native-v2 baseline. This does not require old native-v1 training values to match.

Three alternating same-host serial probe pairs on the final source gave these medians and process high-water memory values (each process runs the four combinations in the displayed order):

| Combination | Before seconds | After seconds | Before peak MiB | After peak MiB |
| --- | ---: | ---: | ---: | ---: |
| PPO Base | 0.2560 | 0.2575 | 303.1 | 303.7 |
| PPO SL | 0.3411 | 0.3412 | 314.7 | 314.9 |
| DQN Base | 0.2465 | 0.2493 | 316.5 | 316.6 |
| DQN SL | 0.4570 | 0.4581 | 331.9 | 331.8 |

All six processes produced identical action/reward/model results. These are short noisy engineering probes, not throughput guarantees or research comparisons. Median time differences were approximately 0–1.1%; these samples do not establish a reproducible performance regression.

## Archive and limits

The archive checksum was verified. Its source can construct and load a representative initial pre-component PPO checkpoint, restoring tensors, RNG and counters under the preserved implementation. This is an archive-loader smoke check, not evidence that every historical run has sufficient continuation state. The earlier historical model-only package inspection is recorded separately in the archive index.

The new component checkpoint contract rejects older contracts. YAML migration creates new experiments and never converts checkpoint weights or state. No historical `runs`, `artifacts` or frozen source was moved. Original archived environments were not all rebuilt; CUDA/H20, Windows and real W&B cloud connectivity were not qualified. Formal research runs remain outside this change.
