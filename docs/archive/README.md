# Historical source and recovery evidence

`index.json` records source archive hashes and evidence locations relative to the
original repository. The large tar files are local under
`backup/2026-10-28-runtime-consolidation/`; they are deliberately not committed.
Existing runs, artifacts and frozen-source directories have not been moved.
A fresh clone needs those local archives/evidence supplied separately.

## Reconstruct source

1. Select source from the run's manifest, observation identity, extension hashes
   and dependency record. A more recent source snapshot is not a substitute.
2. Verify the selected tar's SHA256 against `index.json` before extraction.
3. Use a separate recovery checkout at the recorded commit, then extract the
   matching snapshot into it. Retain `.git`: a plain tar extraction alone has no
   commit identity and may correctly fail strict continuation checks.
4. From that checkout use its Python version, `pyproject.toml` and `uv.lock`.
   Install the extras required by that historical runtime. Rebuild the matching
   platform/environment where strict continuation requires it.
5. Inspect the archived README and original command. Current CLI/identity rules
   do not migrate historical checkpoints. Keep source and result backups; do not
   disable identity checks or edit manifests to force acceptance.

For example, after selecting an archive and an unused recovery path:

```sh
# Run from the original Git repository; use values from index.json.
git worktree add --detach /absolute/recovery/source RECORDED_COMMIT
tar -xf /absolute/backup/SELECTED_ARCHIVE.tar -C /absolute/recovery/source
cd /absolute/recovery/source
uv sync --locked --extra learning --extra cpu
```

This reconstructs a candidate environment, not proof of resumability. The archive
contains the historical lock, but this change does not certify every historical
platform installation or checkpoint's optimizer/replay/RNG completeness.

## Evidence levels

| Evidence | Verified in this change | Boundary |
| --- | --- | --- |
| Pre-consolidation main and social source tar | SHA256, source/tests/lock availability | Reconstruction instructions; no blanket checkpoint compatibility |
| Historical grid/spatial recordings | Current read-only readers and public playback route | Reading does not rerun old physics |
| Profiling-handoff PPO component | Matching preserved dirty source loads metadata, encoder and 16 tensors | Evaluation not run; complete continuation not established |
| Same component with pre-consolidation main | Correctly rejects public-field digest mismatch | Do not bypass compatibility checks |

The PPO component points to an existing frozen source based on `107592b4` with
16 modified tracked files. Its aggregate content hash and exact changed paths
are indexed. This is why a Git commit alone cannot describe that historical run.
The model package contains no complete optimizer/replay/training-RNG transaction.

`retired-tests.json`, `retired-configs.json` and `retired-entrypoints.json` identify
withdrawn implementations by original bytes. Mixed tests retain current shared
contracts; archived implementation tests must execute in their own source tree.
Historical ADRs and result records keep their original meaning.
