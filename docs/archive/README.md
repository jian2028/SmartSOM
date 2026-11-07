# Source archive and recovery

Only the archive index and this recovery note are maintained here. Historical
executable sources, configuration copies and validation reports are outside the
tracked tree. Existing runs/checkpoints and Git history were not modified.

The `before-unified-observation-components` entry in `index.json` identifies a
complete pre-change source archive, including uncommitted/new files, uv.lock,
experiment configs and fixed test panels. Its external `archive-manifest.json`
records each file's SHA-256 and the original HEAD; `pre-change.patch` records the
tracked dirty changes. Extraction into an independent baseline was verified
against every manifest hash (1,017 files).

To recover, copy that archive and manifest to a separate directory, verify the
archive SHA-256, extract it, and verify every file against the manifest. Use the
archived lockfile/environment and the original run's frozen extensions and source
identity. Source restoration alone does not establish that every old checkpoint
has been resumed. Do not copy historical implementations back into current main.

The new Base/SL recipes use native initialization. Historical checkpoints,
standalone workload inputs and recordings use their archived reader; they are not
silently converted or relabeled. Other archive entries retain their recorded
provenance; their availability is not re-certified by this cleanup.

The verified archive is stored outside all Git worktrees at
`/Users/jianni/SmartSOM-archives/20261106-unified-components`. Local benchmark
restorations and validation reports are separate ignored evidence, not a second
maintained runtime.
