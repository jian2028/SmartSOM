# 0093 — Validate the runtime against Ray 2.60.0

Status: user-authorized upgrade; acceptance pending.

This supersedes only the Ray 2.58 version restriction in ADR0089. The user
authorized moving to the latest stable release, verified as Ray 2.60.0, after
reviewing DQN integration failures. Keep the new API stack, official exploration,
algorithm losses and previously approved scientific contracts unchanged.

Install into an isolated test environment. Preserve the Ray 2.58 environment,
frozen sources, benchmark evidence and existing experiments. Do not upgrade any
running experiment's environment or promise cross-version checkpoint recovery.

Release notes alone do not establish that our failures are fixed. Revalidate
official replay persistence, replay restoration and eviction, legal action masks,
resource declarations, fixed evaluation and official checkpoint selection. Keep
approved checkpoint extensions only where still needed and verified. The known
zero-Q exploration boundary does not authorize custom epsilon-greedy.

The previous 2.58 performance results remain historical evidence. New acceptance
compares direct RLlib/Tune with SmartSOM using the same locked 2.60 dependencies
and workload; the agreed five-percent limits remain unchanged. No publication,
merge, server migration or formal research run is authorized by this upgrade.

Static comparison of the installed 2.58 and 2.60 implementations confirms that
new-stack Algorithm state still omits replay, and MARL replay restore still
omits the inverse eviction index and uses incorrect missing-counter/minimum-tree
defaults. Retain the authorized checkpoint extensions and test real restoration
and continued eviction on 2.60. The release's replay metrics fix does not repair
these state-restoration paths.
