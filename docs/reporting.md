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
    export_model,
    export_experiment,
    import_bundle,
    verify_bundle,
    locate_reference,
    model_locator,
    relocate_reference,
)

export_model(run, "exports/model.zip")
export_experiment(run, "exports/experiment.zip")
verify_bundle("exports/experiment.zip")
imported = import_bundle("exports/experiment.zip", "imports/experiment")
model = model_locator(imported, checkpoint="last")
```

A model export contains the actual framework files, original checkpoint manifest,
and a relative `checkpoint_algorithm.json` descriptor. When available, its original
`resolved_training.json` is included beside the checkpoint directory. An experiment
export contains the source evidence tree and explicitly declared external model
inputs. Internal symlinks are materialized as actual
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
For an evaluation-only run, the exporter reads checkpoint identity `path` and
`training_snapshot` fields from v2/evaluation metadata and evaluation plans. It
also recognizes checkpoint fields in algorithm descriptors and resolved run
snapshots. These declared external files are included under `_dependencies/`,
deduplicated, and checked against available checkpoint/snapshot digests. Other
external authoring paths, arbitrary JSON strings and escaping symlinks are not
dependencies. No adjacent source tree or unrelated training output is copied.

Readers use `locate_reference(owner_metadata_path, historical_value)` to resolve
these inputs. Inside an imported package, historical absolute paths always use
the relocation map, even if the original source still exists. Re-exporting an
imported evaluation preserves this behavior. References not captured in the
package remain unavailable. A model bundle
is self-contained for its model files; dependency installation and input-structure
compatibility still apply. Historical exports are inference checkpoints unless
their own manifest explicitly provides supported training-resume state.

`model_locator` understands historical final checkpoints, explicit checkpoint
descriptors, v2 training roots, and `checkpoints/{last,best}.json` selection files.
For a new update checkpoint it verifies the complete update inventory and returns
`inference/`. `best` never silently falls back to `last`. The complete experiment
export retains update training state; a model export includes inference files and
the training recipe, without claiming that the model ZIP supports training resume.

## Offline report and publication figures

```python
from smartsom.experiments.report import build_report, export_figure

build_report(run, "exports/report.html")
export_figure(run, "exports/timeline.png")
export_figure(run, "exports/timeline.svg", resource="machine:M1")
export_figure(run, "exports/timeline.pdf")
```

HTML embeds its data, SVG renderer and JavaScript; no server, CDN, account or
network request is needed. Open the file in a browser. Select the run or retained
failed episode, filter by resource or job, and play/pause, step, or scrub through
the original event order. Events at the same tick remain separately selectable.
The event table shows the latest 120 matching records at the cursor; moving the
cursor exposes earlier records without dropping them from the report.

Processing and AGV travel/wait intervals come from actual recorded transitions.
Machine pauses do not count as processing, and incomplete intervals are marked.
Training views expose raw returns, completed-episode makespans and persisted
learner metrics; absent makespans remain absent. A successful training ledger is
not relabeled a physical trace. Playback is a display of recorded evidence, not
an invocation or replacement of simulator replay validation.

The HTML toolbar can save the current Gantt as SVG or PNG and use the browser's
Print / Save PDF feature. `export_figure` writes PNG/SVG/PDF using the optional
`reports` extra and a headless matplotlib backend. For a multi-run input its
static default is the first recorded timeline; pass the report's `run_id` to
choose another. Training-only inputs use the first recorded learning curve.
Figures keep semantic resource IDs and job colors; hatching distinguishes empty
travel, unloading waits, downtime and partial evidence.

Reports write outside historical evidence. A new v2 experiment may keep derived
reports in its own `reports/` directory. Existing output files are never replaced.
The default HTML limit is 100,000 trace records, and a JSONL input is capped at
256 MiB; exceeding a bound fails explicitly instead of silently truncating the
experiment. Select a narrower child run when needed.
