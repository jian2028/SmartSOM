# Scenarios and Studio

Use one `scenario.yaml` for the production world, one Algorithm for policy
components, and one Experiment as the launch entry. Current shared scenes are
`configs/scenarios/small.yaml` and `configs/scenarios/medium.yaml`. A study can
put its scene in `common/scenario.yaml`; variants reference that same file.

```yaml
schema: smartsom.base-experiment/v1
scenario: ../common/scenario.yaml
algorithm: algorithm.yaml
train_seeds: [101, 102, 103]
```

This is an input-reference example; actual training still requires the existing
budget/calibration settings. Use the checked recipes under `experiments/examples`.

## Machine and vehicle defaults

Within a scene's `factory` section:

```yaml
machines:
  defaults:
    processing_rate_multiplier: 1.0
    quality_modes:
      - quality_mode_id: normal
        time_scale: 1.0
        error_rate: 0.018
    # breakdown: enabled, intervals.low/medium/high, repair
  items:
    - id: machine_00
      operation: A
      position: {x: 3, y: 1}
    - id: machine_01
      operation: B
      position: {x: 8, y: 1}
      overrides:
        processing_rate_multiplier: 0.8
agvs:
  defaults:
    job_capacity: 1
    move_cells_per_tick: 1
    initial_heading: east
    battery: null
  items:
    - id: agv_00
      position: {x: 3, y: 0}
```

These are excerpts; the full scene includes grid, buffers, ports, job rules and
authoring metadata. Machine IDs, operations and positions are always explicit.
Do not specify machine count, size or rotation. Count comes from the items list;
machines and vehicles are 1×1 cells. Maps with non-1×1 machines cannot be migrated
without a deliberate geometry change. Other physical objects retain their size.

`factory.buffer_defaults.machine_pre.storage` and `machine_post.storage` supply
Pre/Post defaults. A buffer's explicit storage fields override its role default.
Changing a default affects only inherited values. A deliberately written equal
value is still an override. Lists (including quality-mode tables) replace whole
lists. For nested mappings only specified fields override, except a change of
storage `mode`, which replaces that storage definition.

## Job rules and failures

The scene's `jobs` section groups:

- `orders.total` and `orders.initial`;
- `arrivals.mean_interval_ticks`;
- `routes.families` and `repeat_first_probability`;
- `processing.base_ticks` and `relative_variation`;
- `due_dates.urgent_probability`, `normal_multiplier`, `urgent_multiplier`.

No generation law changed. Route families are sampled uniformly. Processing work
varies uniformly around the base by the relative fraction. Due multipliers apply
to the existing processing-plus-transport baseline, so changing a map can change
due dates. The current runner still requires A/B/C/D and one capability per machine.

Fault defaults live under `factory.machines.defaults.breakdown`, with `enabled`,
`intervals.low/medium/high` and `repair`. Each distribution uses `scale`, `lower`,
`upper` in ticks. Scale is the exponential scale before conditional truncation,
not the realized mean. Machine overrides may disable faults or change selected
fields. Experiment conditions remain `normal`, `low`, `medium`, `high`; normal
always disables faults. Fault edits do not draw from the orders random stream.

## Studio workflow

```sh
smartsom studio configs/scenarios/small.yaml
smartsom check current-smoke --profile local
```

Choose Edit, then **Scenario…** for shared device defaults, fault distributions,
and the Job generator. Select a machine, AGV or Pre/Post buffer and choose
**Defaults / overrides…**: unchecked groups inherit; checking a group pins its
values. Existing partial YAML overrides retain unspecified inherited fields when
edited.
Restore defaults clears the overrides. Changing machine ID, operation or position
continues through its property controls. Machine resize handles are absent.

**Case preview** accepts a seed, fault condition and drain horizon. It displays the
world digest, order/fault counts and the first 100 rows of each. This generates
evidence in memory, not a training or simulation run. It uses the same generator
as execution and reports unsupported factory contracts before generating cases.

Apply is one undoable transaction. Save and recovery retain defaults, overrides,
job rules and drawing state. YAML formatting is regenerated; handwritten comments
are not preserved. New current templates are Small/Medium; older shape templates
remain source evidence and renderer fixtures, not new-scene menu choices.

## Migration and frozen results

```sh
smartsom config migrate path/to/old/experiment.yaml --output path/to/new-experiment
smartsom check path/to/new-experiment/experiment.yaml
```

For current Algorithm-component experiments, migration writes Scenario, Algorithm,
Experiment and a report. Original files remain unchanged; an existing destination
is rejected. Known workload matrix paths are rewritten; ambiguous factory matrix
edits require explicit authoring. Existing case/checkpoint/calibration references
remain tied to their original files. Migration creates new experiment definitions,
not compatible checkpoints. The older Algorithm migration remains a separate
explicit step for older Algorithm schemas.

Scenario matrix axes use the same compiler as a single experiment, for example
`scenario.jobs.orders.total: [128, 256]`. Input references such as
`experiment.scenario` are not scientific matrix axes. Migration also copies
custom components used only by matrix variants, including declared `source_root`
bundles and resources, so removing the original extension files does not break
the migrated definition.
When varying `source_root` changes an entry's bundle scope, use complete component
declarations as matrix values (including `path`, `factory`, and `source_root`).
A scalar `source_root` axis that would require a correlated path rewrite is
rejected before creating output, with this migration instruction.

Opening a bare legacy factory cannot recover its old job rules. It imports geometry
with the documented default job generator; migrate the full Experiment to preserve
the paired workload. Legacy factory `reliability` is rejected because it is not
the current workload fault protocol. Explicit four-file inputs still compile to
the same internal contracts; new examples use scenes.

Runs retain both `inputs/authoring/scenario.yaml` and expanded `factory.yaml` /
`workload.yaml`, with Algorithm and Experiment inputs. Expanded files are generated
evidence, not additional files users must maintain. Resume uses frozen inputs;
it does not reread a modified authoring scene. Old runs/artifacts stay in place.
