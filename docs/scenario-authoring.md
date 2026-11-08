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

## Frozen inputs and historical source

Scenario is the only current human-authored scene input. `scenario.jobs.*` and
`scenario.factory.*` are the explicit override paths. Standalone workload files,
old Factory/Workload experiment references and `config migrate` are retired.
Internal workload/geometry models remain necessary for deterministic generation.

Compilation preserves the authored Scenario and writes a
`smartsom.resolved-experiment/v1` snapshot with expanded runtime data. Current
resume validates that frozen content and component/source identities. Recompiling
a current snapshot uses its preserved authored Scenario. Historical snapshots,
checkpoints and recordings require the archived version; see the archive index.
