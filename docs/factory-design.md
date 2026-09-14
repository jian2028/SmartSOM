# Factory design v2

This is the complete factory authoring format for SmartSOM Studio. The current
runtime still consumes `smartsom.factory/v1`; the two formats are intentionally
separate. Previewing or validating a v2 file never runs a simulation.

## File envelope and names

```yaml
# Coordinates are grid cells; time is ticks; energy uses abstract units.
schema: smartsom.factory/v2
factory:
  factory_id: factory_001
  name: Untitled Factory
  grid:
    width: 20
    height: 15
    blocked_cells: []
  machines: []
  buffers: []
  inspection_stations: []
  scrap_bins: []
  chargers: []
  ports: []
  agvs: []
```

YAML keys and Python functions/modules use snake_case. Frozen Python types use
`FactoryDesign`, `MachineDesign`, `BufferDesign`, `PortDesign`, `AGVDesign`,
`ChargerDesign`, `InspectionStationDesign`, `ScrapBinDesign` and `SlotDesign`.
Qt graphics classes use the corresponding `Item` suffix.

Resource IDs are globally unique within a factory, case-sensitive and match
`[A-Za-z][A-Za-z0-9_-]*`. Generated IDs use full type prefixes and at least three
digits, such as `machine_001` and `inspection_station_001`. IDs are never list
indices. Slot IDs are unique within their owning buffer/station; quality mode
IDs within their machine. Each top-level resource has a nonempty Unicode `name`.
Duplicate display names are permitted. Naming does not establish resource identity.

## Geometry

Cell coordinates are `{x: integer, y: integer}`. The origin is the top-left cell;
x grows right and y grows down. Grid dimensions are positive integers. Obstacles
are unique `grid.blocked_cells`, not named objects. There are no metres-per-cell.

Solid resources have `footprint: {x, y, width, height, rotation}`. Dimensions are
positive integers and describe current occupied cells. Rotation is clockwise
`0`, `90`, `180` or `270`; it describes orientation and must not be reapplied to
dimensions or slot positions when loading/rendering.

For the later rotation command, a 2×3 footprint at (10,5) becomes 3×2 at (10,5).
A slot at local (0,2) becomes local (0,0), retaining its ID. Clockwise local
transformation is `(u,v) -> (old_height-1-v,u)`; rotation metadata increases by
90 modulo 360. World coordinates always equal parent origin plus local cell.
Group rotation uses the same transformation around the selection's top-left
bounding box; group relocation must validate atomically.

Machines, buffers, inspection stations, scrap bins and chargers are solid:
they cannot overlap one another or blocked cells. Slots are internal to their
parent footprint, unique by cell and ID, and may leave unused cells. Ports must
be walkable and occupy distinct cells. AGV initial cells must be distinct and
walkable, and may coincide with ports. Footprints, ports, AGVs and obstacles
must stay inside the grid.

## Resource fields

All solid resources also have their typed ID, `name` and `footprint`.

| Resource | Fields and interpretation |
| --- | --- |
| Machine | `machine_id`, `operation_types`, nonempty `quality_modes`; one concurrently processed job regardless of footprint. |
| Quality mode | `quality_mode_id`, finite decimal `time_scale > 0`, finite decimal `error_rate` in [0,1]. Normal is 1/0. Base operation durations belong to workload, not this file. |
| Buffer | `buffer_id`, `role`, `storage`, nullable `machine_id`. |
| Slot | `slot_id`, `local_cell`, positive integer `capacity`, default 1. |
| Inspection station | `inspection_station_id`, `slots` (capacity exactly 1 each), positive integer `inspection_ticks`, `parallel_capacity` as positive integer or `max`. Inspection reveals existing quality accurately; no false positive/negative parameters. |
| Scrap bin | `scrap_bin_id`, nonnegative integer or null `capacity`; accepts confirmed defects as terminal drop-offs, not retrievable inventory. |
| Charger | `charger_id`, positive integer `agv_capacity`, positive finite decimal `charge_energy_per_tick` per AGV, not a shared power budget. Its body is solid; charging ports are separate walkable cells. |
| AGV | `agv_id`, `name`, `initial_cell`, `initial_heading`, positive integer `job_capacity` and `move_cells_per_tick`, optional `battery`. Footprint is exactly one cell. |
| Battery | Positive finite `energy_capacity`, `initial_energy` in [0,capacity], nonnegative finite `move_energy_per_cell` and `idle_energy_per_tick`. |

`operation_types` contains distinct semantic processing-category IDs, initially
`operation_1` through `operation_4` (displayed as Operation 1–4). A machine may
support multiple categories, and multiple machines may share a category. IDs
follow the ordinary identifier syntax and are not restricted to these four.
This is separate from quality/speed modes, port pickup/drop-off operations, and
the concrete operations in a job's workload. Job precedence, machine alternatives
and base processing times remain workload responsibilities. These editor
declarations do not yet drive simulator eligibility.

An omitted `operation_types` field loads as an empty tuple: **unspecified**, not
all categories or no processing capability. No category is inferred from a
machine ID, name or position. Older files remain readable; saving writes the field.

Decimals are normalized and saved as exact decimal strings. Boolean values are
not accepted as integers. `battery: null` means energy is not modelled; new AGVs
default to no battery, load capacity 1, speed 1 cell/tick and heading east.
Battery defaults when explicitly created are capacity 100, initial energy 100,
move consumption 1/cell and idle consumption 0/tick. Charger defaults are one
AGV and 10 units/tick per AGV. Inspection defaults are 2 ticks and `max`.

Buffer roles are `storage`, `machine_pre`, `machine_post`, `system_input` and
`system_output`. Each machine may have neither, one or both dedicated buffers;
each side has at most one. `machine_id` is only applicable to pre/post roles.
The machine stores no duplicate reverse references. Each system input/output
role can occur at most once. Public storage is independent of machines.

```yaml
storage:
  mode: pool
  capacity: null  # Unlimited; 0 and positive integers are also valid.
```

```yaml
storage:
  mode: slots
  slots:
    - slot_id: slot_001
      local_cell: {x: 0, y: 0}
      capacity: 1
```

Pool capacity is independent of footprint area. Slot-storage capacity is the sum
of slot capacities; it is not stored twice. Empty slot collections are allowed
as unfinished designs. Inspection `max` means all currently declared slots;
an explicit parallel limit greater than available slots produces a warning.

## Ports and bindings

**One port is one AGV interaction cell**, not a group of child docks. Multiple
interaction positions are multiple ports. Fields are `port_id`, `name`, `cell`,
nonempty `allowed_headings` (north/east/south/west) and `bindings`.

```yaml
port_id: port_001
name: Pre M1 access
cell: {x: 7, y: 3}
allowed_headings: [north, east, south, west]
bindings:
  - target: {kind: buffer_slot, buffer_id: buffer_004, slot_id: slot_001}
    operations: [pickup, drop_off]
  - target: {kind: buffer_slot, buffer_id: buffer_004, slot_id: slot_002}
    operations: [pickup, drop_off]
```

| Target kind | Required identity | Operations |
| --- | --- | --- |
| `machine` | `machine_id` | pickup, drop_off, or both |
| `buffer` | `buffer_id` pointing to pool storage | pickup, drop_off, or both |
| `buffer_slot` | `buffer_id`, `slot_id` | pickup, drop_off, or both |
| `inspection_slot` | `inspection_station_id`, `slot_id` | pickup, drop_off, or both |
| `scrap_bin` | `scrap_bin_id` | drop_off only |
| `charger` | `charger_id` | charge only |

Each target occurs once per port; operations cannot be empty or duplicated.
Multiple ports may access the same target. Binding is authoritative: there is
no inferred adjacency, heading-to-target or maximum reach constraint. Moving a
resource alone keeps ports in place and their ID references intact. Geometry
does not introduce a port-level service lock or implement vehicle concurrency.

## Diagnostics and files

Constructors enforce local field types/ranges. `validate_factory_design()`
returns immutable `DesignIssue` entries with severity, code, message, entity ID
and field. The file reader and writer reject errors, including invalid geometry,
duplicate IDs and dangling references. Empty ports, unassigned pre/post owners
and empty slot facilities are draft warnings. An empty factory or a machine
without buffers is valid, not an unfinished buffer assignment.

`config.factory_design` exposes:

```python
design, byte_digest = load_factory_design(path)
issues = validate_factory_design(design)
new_digest = save_factory_design(new_path, design)
new_digest = save_factory_design(path, design, expected_digest=byte_digest)
```

Files use `.yaml` or `.yml`; unknown keys, duplicate keys and incorrect schemas
are rejected. The reader obtains data and digest from the same byte snapshot.
Saving without a digest is create-only. Replacing an existing file requires its
last observed digest. External changes/removal require reloading or Save As.
Same-directory temporary publication avoids partial files; digest comparison
is optimistic, not a cross-editor locking guarantee.

Saving retains all design data with regenerated formatting and a few fixed
system comments. Original handwritten comments, aliases and key layout are not
preserved. A template is the same complete YAML, without additional factory
metadata hidden in Studio settings.

## Template 1 (default)

The compact `studio/templates/template_001.yaml` is a 12×12 design with identity
`factory_001` / `Template 1`. Four machines are named Machine 1–4 in row-major
order, each initially supporting the corresponding Operation 1–4 category.
Categories are independently editable; neither numbering nor position declares
a job route. Dedicated buffers and ports are named after their physical machine.

| Facility | Top-left | Size | Parameters |
| --- | --- | --- | --- |
| Machine 1 / 2 | (2,0), (8,0) | 2×2 each | Operation 1 / 2 respectively; Normal quality, scale 1, error 0 |
| Machine 3 / 4 | (2,10), (8,10) | 2×2 each | Operation 3 / 4 respectively; same quality |
| Input / Output | (0,5), (11,5) | 1×2 each | Unlimited pools |
| Inspection | (4,5) | 4×2 | 8 slots, 2 ticks, parallel max |
| Scrap bins | (3,5), (8,5) | 1×2 each (vertical) | Unlimited, drop-off only |
| Chargers | (0,4), (0,7), (11,4), (11,7) | 1×1 each | One AGV, 10 units/tick each |

Every machine has a 2×2 pre-buffer directly left and a 2×2 post-buffer directly
right, each with four capacity-one slots. Machine IDs are 001..004; buffer IDs
001/002 are Input/Output and 003..010 are paired pre/post buffers. Group origins
are (0,0), (6,0), (0,10), (6,10). Each group has two ports directly in front of
the machine, at group origin x+2 and x+3, on y=2/9 toward the central aisle.
The left port binds the pre-buffer, the right the post-buffer; each reaches all
four slots in its buffer with pickup and drop-off. Port IDs and bindings are
unchanged by the relocation.

Input ports are (1,5)/(1,6), pickup only; Output ports are (10,5)/(10,6), drop-off
only. Each pair occupies adjacent cells beside its buffer, toward the map interior.
Inspection ports at x=4/5/6/7, y=4/7 each bind the nearest same-column slot with
both operations. Each scrap bin has two ports directly above and below it:
x=3 on the left and x=8 on the right, y=4/7, all drop-off only. Charger ports
are (1,4)/(1,7) on the left and (10,4)/(10,7) on the right, charge only.
Four fully charged AGVs start at (3,4), (8,4), (3,7), (8,7),
facing inward (east on the left, west on the right).

The machine groups touch in the middle of each row, with no central charger column.
Each side has a charger directly above and below its two-cell Input/Output buffer.
Input shares the first column of the left pre-buffers; Output shares the last
column of the right post-buffers. Inspection is centered on both map axes, with
the scrap bins touching its left and right edges: together they fill a centered
6-column by 2-row rectangle at (3,5), centered on the even-width map. The four AGVs
initially occupy the scrap ports.
All resource footprints, port cells and AGV starts are mirrored across both axes.
There are 28 ports listed in (y,x) order, 68 solid cells and 76 connected walkable
cells. Resource IDs, slot IDs and retained ports' binding targets are stable.
The added inspection slots slot_007/008 occupy its rightmost column, served by
port_031/032. Earlier slots and their binding identities are retained. The left
scrap bin is scrap_bin_002, served by port_025/027; the right is scrap_bin_001,
served by port_009/012. The extra former side ports 010/013/026/028 are removed;
remaining port IDs are not reindexed by the new positions.
Chargers 001/002 and ports 001/022 are relocated to the left side; new chargers
003/004 use ports 029/030 on the right. Former port IDs are not recycled.

## Template 2

The bundled `studio/templates/template_002.yaml` preserves the former eight-machine
map with identity `factory_002` / `Template 2`. It is a 42×12 symmetric example.
Top groups occupy y=0..1, bottom groups y=10..11. For each row, group origins
are x=6,14,22,30; each group consists of adjacent 2×2 pre, machine and post.
Machine IDs 001..004 run across the top; 005..008 across the bottom. Buffer IDs
001/002 are Input/Output; 004..019 are paired pre/post buffers. The former
shared buffer 003 is replaced by `inspection_station_002`; other IDs are retained.

| Facility | Top-left | Size | Parameters |
| --- | --- | --- | --- |
| Input | (0,4) | 2×4 | Unlimited pool |
| Output | (40,4) | 2×4 | Unlimited pool |
| Inspection 1 | (14,5) | 4×2 | 8 slots, 2 ticks, parallel max |
| Inspection 2 | (24,5) | 4×2 | 8 slots, 2 ticks, parallel max |
| Scrap bin | (20,5) | 2×2 | Unlimited |
| Chargers | (3,0), (38,0), (3,11), (38,11) | 1×1 each | one AGV, 10 units/tick each |

Every pre/post buffer has four slots and one port reaching all four. Their port
x positions are group origin+1 and +4, with y=2 for the top row and y=9 for the
bottom row. Both pickup and drop-off are enabled. Input ports are (2,4)/(2,7),
pickup only; output ports are (39,4)/(39,7), drop-off only. Both access their pool.

Inspection 1 ports have x=14..17 and y=4/7; Inspection 2 ports have x=24..27 and y=4/7.
Each binds its same-column nearest slot and supports both operations. Scrap
ports use x=20/21 and y=4/7, drop-off only. Charger ports use x=3/38 and y=1/10,
charge only. Four AGVs start on those charger ports; top AGVs face north, bottom
AGVs south. Batteries are enabled with the defaults above. All port headings
are permitted. Ports are numbered in (y,x) order; charger ports are 001/002/043/044.

There are 44 ports, 136 solid cells and 368 walkable cells. Solid and port
geometry is mirrored across both map axes. Machine groups touch the top/bottom
boundary, leaving three walkable rows toward the central facilities. Chargers
also touch the top/bottom boundary; Input/Output touch the left/right edges. Static
connectivity does not establish multi-AGV collision or deadlock properties.
