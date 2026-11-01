# Network ownership and research-package engineering acceptance

Scope: the supplement after `67cc411`, reviewed as immutable dirty-source snapshot
`e30a116d1a4d9a8a80f56396c0e16fa7aed2cbfd23bdcb9ba3ad7099fded9686`,
with the final explicit-seed precedence delta independently reviewed in snapshot
`d838acf52944ba10295b61bd09296c026e31ec402ed1094ead3bb720b1743673`.
These are bounded engineering checks, not research results. No formal campaign
was launched, and no historical run directory was moved or rewritten.

## Verification

- Ruff check/format and diff whitespace checks passed.
- Full unit/package plus affected ownership, exact-node and lifecycle suite:
  1,455 passed, four documented platform-only skips. The subsequently added real
  one-sided imported-panel evaluation test also passed. Final configuration/seed
  precedence regression: 49 passed.
- The other integration files ran in separate processes: 196 passed, including
  Gym, native extensions, scheduler, W&B/TensorBoard, and offscreen Studio/replay.
  This count overlaps the focused suite; it is not an additional unique-test total.
- Six local Ray/Tune tests passed, including two-seed runs, pause/resume and actor
  replacement. Actor replacement was rerun against unchanged source after an
  earlier development run correctly rejected a source edit during continuation.
- Frozen peer-pressure OFF and ON packages each completed 160 physical ticks,
  15 learner updates, model save and full learner restore. Temporary probes and
  their outputs are outside the repository.
- The three migrated packages compile to 12 peer-pressure, six owner-sharing and
  27 Base/SL trials. Compilation did not launch them.

Independent engineering, scientific and test reviewers examined full callers and
new files, then reviewed the fixes. Confirmed findings were closed: DQN target
clock changes from redundant role overrides; missing training-world fields in
SIG identity; imported/generated validation-test overlap; wrong model paths in
reports; and the mixed-panel import error introduced during repair.

The non-implementer designed and executed five-role PPO/DQN ownership and full-state
continuation probes, actual 0/7/19-tick validation, and a complete two-seed batch
stop/copy/CLI-resume with the old path made unavailable. Existing case receipts
were preserved. Same-platform resume leaves prior manifests byte-identical.
Original 20-machine Studio document load/save/reload preserved the full design.

## Same-machine 160-tick differential

Each row uses a fresh process, fixed inputs/actions/RNG, one rollout and update.
Baseline is `67cc411`; all four trace, physical-reward and updated-model digests
matched exactly. These single short timings do not establish a stable speedup.

| Combination | Before seconds | After seconds | Before peak RSS bytes | After peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| Base PPO | 0.2524 | 0.2528 | 320389120 | 320864256 |
| SL PPO | 0.3738 | 0.3712 | 334233600 | 333578240 |
| Base DQN | 0.2591 | 0.2576 | 321470464 | 321142784 |
| SL DQN | 1.1526 | 1.1684 | 402161664 | 403570688 |

An earlier same-input Machine-independent PPO probe had 521,904 parameters versus
217,460 shared, 12 versus five optimizer updates, and about 331 MB versus 321 MB
peak RSS. Its subsecond timing is insufficient for a throughput conclusion.
Independent ownership intentionally increases model and optimizer/replay storage.

## Evidence boundaries

Actual hardware validation here is macOS CPU. Hosted CI supplies its own Linux
and macOS results on the final PR commit. No actual cross-OS/CUDA/cloud migration,
mobile browser rendering, or formal algorithm comparison is claimed. The portable
bundle checks move directories on one host. Studio's offscreen UI suite and original
map document roundtrip were checked; no manual desktop editing session was run.

Historical migration limitations and explicit protocol additions are documented in
`docs/migration/research-packages.md` and each package README. Remote historical
manifests/environments were not fetched; old checkpoint compatibility is not implied.
