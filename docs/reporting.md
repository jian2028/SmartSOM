# Local experiment catalog and portable exports

Each experiment's real directory owns its evidence. The catalog and the
`models/`, `logs/`, and `reports/` navigation views are disposable conveniences.
Historical runs and their manifests remain byte-for-byte unchanged.

```python
from smartsom.experiments.catalog import list_runs, rebuild_views, resolve_run

entries = list_runs(["runs", "artifacts/learning", "artifacts/resource-marl"])
run = resolve_run("YOUR_RUN_ID", ["runs"])
rebuild_views(["runs"], "artifacts/catalog")
```

A v2 experiment has a `run.json` with schema `smartsom.experiment/v2`, string
`id`, `name`, `kind`, and `status`, and a `paths` mapping. All declared paths are
relative to that experiment and must stay inside it. Optional `provider`, `seed`
and `tags` describe the run; other metadata is permitted. Current experiment
roots are indexed once, without separately indexing their child evidence.
Historical training, single-run and study manifest formats are also recognized.
Ambiguous names and ID prefixes require an explicit choice.

The catalog writes only outside the experiments. Its index stores relative paths,
and its links are relative symlinks. Deleting a link does not delete its source.
Rebuilding recreates missing links and removes only stale links previously owned
by that index. Existing regular files and user-modified links are never replaced.
For old runs without separate log directories, the logs link opens the evidence
directory containing `progress.log`.

## Export and import

```python
from smartsom.experiments.packaging import (
    export_model, export_experiment, import_bundle, verify_bundle,
    relocate_reference,
)

export_model(run, "exports/model.zip")
export_experiment(run, "exports/experiment.zip")
verify_bundle("exports/experiment.zip")
imported = import_bundle("exports/experiment.zip", "imports/experiment")
```

A model export contains the actual framework files, original checkpoint manifest,
and a relative `checkpoint_algorithm.json` descriptor. An experiment export
contains the source evidence tree. Internal symlinks are materialized as actual
files; external or cyclic links fail explicitly. Export never follows arbitrary
links outside the selected evidence root. Git metadata, virtual environments and
Python bytecode caches are excluded.

The outer `bundle.json` records each member's size and SHA-256, its original
location, and a relocation map. Import verifies the complete inventory before
extracting into a new destination. Traversal, absolute paths, symlink members,
duplicates, corruption and excessive declared size/count are rejected. Checksums
establish integrity, not the identity or trustworthiness of a model's author.
Import itself never executes or loads framework model data.

Original snapshots and reports keep their exact bytes, including historical
absolute references. `relocate_reference(imported, original_path)` maps an
included reference to its current package location without editing that evidence.
References outside the exported root are not magically available. A model bundle
is self-contained for its model files; dependency installation and input-structure
compatibility still apply. Historical exports are inference checkpoints unless
their own manifest explicitly provides supported training-resume state.
