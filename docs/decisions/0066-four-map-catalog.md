# 0066 — Four standalone factory resources

Status: implemented; engineering checks and independent review are separate evidence.

Supersedes the bundled numbered-template catalog in ADR0014/0015 and the repository
map inventory in ADR0061/0062. Scenario, Algorithm and Experiment remain the main
authoring interface defined in ADR0062. No factory-reference inheritance layer is added.

The repository and installed package contain only four independently stored Factory
YAML files, under `src/smartsom/config/maps/`: `small-wide-factory.yaml`,
`medium-wide-factory.yaml`, `small-factory.yaml`, `medium-factory.yaml`.
Studio, the Base factory loader and direct Factory callers share these resources.
The two compact maps are 22×6 and 27×12; the two wide maps preserve the selected
24×6 and 29×19 inputs byte-for-byte. Filename terminology never infers geometry.

Scenario documents may embed those layouts together with jobs and fault rules,
including the eight Small/Medium × four-fault scenarios. The four-file constraint
applies to independent Factory map files, not Scenario snapshots. Existing valid
scenes, workloads, algorithms and experiments are retained. Frozen execution
inputs and user-created files are not repository map presets.

All other tracked standalone factories are removed, with source commit, path and
SHA256 in `docs/archive/four-map-retirement.json`. Historical dependent inputs are
retained and labelled source-only; they are not redirected to a scientifically
different map. Git restores their exact former inputs. Old run worktrees, results,
and personal Studio template settings are not migrated or deleted.

Special editor and frozen-trace geometries are built from typed resources inside
tests. This preserves regression semantics without storing another YAML/JSON map.
The former numbered catalog acceptance is retired; current tests cover the four
resources and the retained historical editor features separately. Local numbered
overrides remain readable by explicit number, while an unavailable retired original
fails clearly instead of selecting a different map. The New dialog and template
manager offer all four current scenes.

This cleanup does not select a fault regime, regenerate expected traces, establish
research equivalence between layouts, authorize formal training, or reuse a
checkpoint/calibration identity across different maps.
