# 0095 — Preserve DQN replay in same-version official checkpoints

Status: user-authorized; verification in progress.

Supersedes ADR0094's acceptance of empty replay after restoration. The user
clarified that avoiding old-version compatibility must not remove persistence of
experience from a newly trained Ray 2.60 DQN run.

Retain the approved thin Algorithm get_state/set_state extension in the official
checkpoint and the necessary replay restoration corrections documented in
ADR0089. These defects were independently confirmed in Ray 2.60 under ADR0093;
the extensions serve same-version recovery, not old-checkpoint compatibility.
There is no separate model or replay storage system, no replacement loss,
exploration, replay sampling algorithm or scheduler, and no Ray source patch.

Both direct performance baseline and integrated runtime use this same extension.
Keep the public AlgorithmConfig get_state/from_state transport fix for the direct
baseline. Preserve previous source snapshots and evidence without accepting them
as migration inputs. Validate replay contents, priorities, indices, learner state,
official counters, fresh-process continuation through eviction, second restore,
and rejection of missing or corrupt replay. Do not promise partial simulator
restoration or bitwise equivalence to uninterrupted training.
