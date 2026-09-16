# SmartSOM Studio

Studio is a local static factory editor built with Qt Widgets and Graphics View.
It opens and edits complete v2 factory YAML and two bundled templates. Every
document starts in Browse; click Edit to change it. It does not execute the
simulator, train policies or replay episodes.

## Factory operation types

Select the factory and choose **Edit operation types…** to manage its catalog.
Automatic mode grows standard Operation 1, 2, 3… for newly placed machines, with
corresponding default capability selections. Add/Delete switches to Manual;
machine capability checkboxes always come from this catalog. A type can be unused,
but deleting a referenced type requires first changing the listed machines.
Types have stable IDs and no machine-count upper bound. A machine without a
selection is marked **capability needs setup** and can still be saved as a draft.

In Manual mode, placement asks for existing capabilities if the corresponding
type is absent. Add a type first if the catalog is empty. Cancel changes nothing.
Restoring Automatic adds the standard types needed for the current machine count
without changing machine selections or deleting extra types. Catalog and mode
changes use Apply/Cancel and Undo/Redo, and survive save, reopening, templates and
recovery. Cross-document paste preserves machine capabilities and asks before
importing missing types; importing and pasting are one undo transaction.

The same v2 YAML holds the full factory plus a separate `authoring` preference.
There is no Studio-only copy of factory truth. This remains a Factory editor:
workload/scenario/algorithm editing, agent decisions and simulation are not added.
The runtime branch now reads this same factory for capability-based workloads,
grid AGV actions, Buffer ranking and inspection. It has retired the matrix
execution path; see [runtime and playback](production-runtime.md).

## Launch

Use the repository's Python 3.12 and optional locked Qt dependency:

```sh
uv sync --locked --extra studio --inexact
uv run --no-sync smartsom studio
uv run --no-sync smartsom studio path/to/factory.yaml another/factory.yml
```

`uv run --extra studio smartsom studio` is also supported. `--inexact` on sync
preserves other optional packages already installed in your local environment.
No learning framework is needed. Help remains available without Qt installed.
The first release is launched from the project command, not a packaged macOS app.

## Inspect a design

Choose New to open Blank (20×15), Template 1 (12×12, four machines, selected by
default), Template 2 (42×12, eight machines), or a v2 YAML as a template.
Template use creates a separate document without changing its source. Open loads
existing files into independent tabs; reopening the same path focuses its tab.
Invalid files produce an error and do not replace existing tabs.

The English workspace has an object tree on the left, a central grid view,
properties on the right, and a collapsible problem list below. Select
an object in the map/tree to inspect IDs, dimensions, capacity, slots, quality
or energy settings. Selecting a problem locates its associated object.

The toolbar switches between **Standard** (resources, map and properties side
by side) and **Focus map** (wide map with the same properties below).
Every launch starts in Standard, even if the previous session ended in Focus map.
In Focus map, **Resources** temporarily shows or hides the object list. The
status-bar **Design checks** button expands or hides the problem list; Focus map
starts with it closed. **Show numbers / Hide numbers**, next to Fit Map, toggles
all resource numbers on the current map in one click. Numbers start hidden in
every new or opened document; each open tab keeps its own visibility choice
when switching tabs or layouts. Grid, numbers and ports are also under **Layers**.
The View menu also exposes layout, layer and problem-list controls.

Switching layouts keeps every open document, selection, property expansion and
slot highlight. Fit Map follows the available viewport size; manual zoom/pan
keeps its scale and center as far as the map bounds allow. Each layout remembers
its panel sizes. Normal application exit saves both layouts' panel sizes
in local Qt user settings (`SmartSOM` / `Studio`), separately from factory YAML.
The resource list starts closed on first entering Focus map in a new session. These preferences
do not save or restore factory documents.

Use the mouse wheel to zoom around the cursor, middle drag or Space-drag to pan,
and Fit Map to restore the full layout. View controls show/hide grid, numbers and
ports. Bindings can highlight those for the selected resource or port (the default),
all ports or none. Selecting a port highlights its exact target slots or the full
footprint of a whole-resource target. Selecting a target resource highlights its
bound slots and associated ports. Shared targets are highlighted once; highlighting
a shared port does not expand the selection into its other targets. A pale yellow
fill expresses declared access, preserving the original grid and facility borders
without extra outlines or connecting lines. Slot selection in the properties uses
the same fill-only highlight. Related ports retain their original blue dashed border.
Port markers represent individual cells. The picture contains design resources,
not simulated jobs or movement.
When enabled, map labels show only each resource ID's numeric suffix; full names
remain in the tree, properties and tooltips. Ports are empty blue dashed cell
outlines. Slot buffers use green grid-aligned edges; each inspection slot has
purple edges and a translucent magnifying-glass symbol. Unused footprint cells
have a neutral fill. Machines use a translucent gear outline, scrap bins a faint
line symbol, and charging stations a translucent lightning-bolt outline without
an arrow.

Input and output pool buffers use a centered tray with a downward receive arrow
or upward dispatch arrow. Both retain the same green fill; the symbol identifies
the buffer's role, not inventory, capacity or AGV heading. PNG/SVG exports use
the same symbols. Other pool buffers and slot grids retain their existing style.

AGVs use a directionless dark teal circular base with a thin white square platform,
without wheels or an arrow. The icon does not rotate with `initial_heading`;
the heading remains unchanged in the design and properties.
Selection adds a pale cyan halo with a blue outer ring around the vehicle, visible
with empty or loaded cargo. The halo does not enlarge the circular click target:
the exposed surrounding port remains independently selectable. Deselecting removes
the halo, and clean map exports do not include selection highlights.
Design previews
start with empty platforms. The AGV graphics item can display a transient loaded
state as a warm clay square (`#B18461`) covering the platform outline.
This symbol means nonempty, not a particular job count. It changes neither
selection nor YAML. No runtime/load-state source or heading-display switch is
connected in this phase; templates do not invent cargo.
Click the circular vehicle body to select the AGV; click the exposed area around
it to select the underlying port. Hover follows the same circular hit area.

**View → Appearance preview…** opens isolated factory and symbol samples. The
manual tick controls show machine and inspection jobs with thin blue or purple
segmented rings. Each fixed segment represents one tick; elapsed segments disappear
clockwise, and at zero the job remains until explicitly cleared. Small white line
icons identify the processing kind inside the job. All job squares use the same
16/40-cell size, including AGV cargo and stored jobs. Idle gears are smaller,
centered outlines without white fill. This preview is a drawing demonstration,
not a connected simulation; it does not modify the document, YAML or map exports.

Buffers keep the same green fill. Their internal slot lines are lighter and
thinner than the single outer outline; touching buffers share one boundary line,
without a gap or an extra frame. Shared boundaries use the stronger of the two
neighbors' states: selected, then hovered, then normal.

Hover a machine, its pre/post buffer, or a bound port **on the map** to emphasize
that machine, its owned buffers and associated ports together. Moving away
restores their normal outlines. A port shared by multiple machines highlights
those directly bound groups, without following other shared ports into further
groups. Public storage and unrelated facilities do not imply machine ownership;
an AGV body above a port remains the topmost hovered resource, while the exposed
port area can be hovered independently.
Hover leaves the selection, properties, slot highlight and binding controls
unchanged. Hidden ports stay hidden. Leaving the view, changing documents/layouts,
deactivating the window or starting navigation clears the temporary emphasis.
These display effects do not modify factory YAML.

Template 1 is the compact default: four machines named Machine 1–4 in two rows,
eight dedicated pre/post buffers, single-column 1×2 input/output aligned with the
machine groups' outer columns, one centered 4×2 inspection station, and two
vertical 1×2 scrap bins touching its left and right edges. Together they form a
centered 6-column by 2-row block. Each scrap bin has access directly above and
below. Four chargers sit directly above and below the input/output buffers;
their ports and the two adjacent ports for each buffer face the map interior.
The former central charger column is removed. Four AGVs start on the four
scrap ports. It has 28 ports, with all footprints, port cells and AGV starts
mirrored across both axes.
Machine 1–4 initially support Operation 1–4 respectively. In Edit, **Supported
operations** provides independent checkboxes: one machine may support several
categories and several machines may share one. Browse shows the selected categories.
Apply/Cancel, undo and YAML saving use the ordinary property workflow. These are
processing categories, separate from quality modes and concrete job operations;
this design does not declare job routes or implement runtime eligibility.
New machines use the catalog defaults described above. Template 2 now declares
eight types and corresponding machine capabilities. Older user files with empty
capabilities remain **Needs setup**; their names and selections are not guessed.
All choices come from the factory catalog, including preserved custom IDs.
Each machine's two ports sit directly in front of its two columns, facing the
central aisle. The left port accesses its pre-buffer, the right its post-buffer.

Template 2 preserves the former eight-machine map: sixteen dedicated pre/post
buffers, input/output, two inspection stations, a scrap bin, four charging
stations, four AGVs and 44 ports. Its resources, geometry and bindings are
unchanged; only the template name and factory ID become Template 2 / factory_002.
See [the complete design contract](factory-design.md) for both layouts' coordinates
and configuration. Existing v1 runtime files are deliberately not imported or
migrated by Studio.

## Edit a design

Click **Edit** beside Browse. The map, resource tree, parameters and checks refer
to the same document. Browse keeps selection, zoom and inspection available but
disables mutations. Switching back to Browse retains applied edits and undo history.

- **Add** offers machines, buffers, inspection stations, scrap bins, chargers,
  ports, AGVs and obstacle tools. Click for a 1×1 resource or drag either way to
  define one rectangular footprint. Ports/AGVs always occupy one cell.
- A new ordinary Buffer uses Slots, one per cell with capacity 1. Inspection
  slots also have capacity 1. Existing template Input/Output pools are unchanged.
- Successful placement returns to **Select**. **Continuous placement** keeps the
  current tool active. Escape cancels an unfinished gesture without consuming IDs.
  Illegal placement appears red, explains the problem and leaves the document intact.
- Click to select; Shift-click adds/removes a resource. Drag on empty space to
  select fully enclosed objects. Drag selected objects to move them together.
  A machine's buffers/ports move only when explicitly selected too.
- Selected solid resources show right, bottom and bottom-right resize handles;
  the top-left corner stays fixed. **Rotate 90°** turns a selection clockwise
  around its bounding box's top-left corner, including local slot coordinates.
- Paint obstacles, draw a rectangular obstacle, or erase obstacle cells through
  Add. Each completed stroke is one undo command. A stroke crossing an occupied
  cell is rejected as a whole.

Parameter forms keep a draft until **Apply**. **Cancel** restores the displayed
values. Apply validates the complete candidate and groups all changed fields in
one undo command. Same-type multi-selection exposes common capability fields;
only edited fields apply to the selected objects. Pool buffers support bulk total
capacity; slot buffers support capacity per existing slot without changing IDs or
holes. Pool/Slots mixed selections require separate storage edits. Editing a
quality-mode table or battery group explicitly replaces that complete group on
the selected resources, as labeled in the form. Mixed selection supports move,
rotate and delete. The factory root exposes its name and grid dimensions.

**Rename ID** is separate from display Name. It updates references atomically.
Generated resource IDs use `machine_001`, `buffer_001`, etc.; slots have stable
owner-local IDs. Deleting a machine keeps its buffers and clears their ownership,
with a design warning. Structural removals show affected slots/references before
confirmation; Undo restores the complete previous state.

### Slots and bindings

**Edit slots** opens a modeless table. Click a footprint cell on the map to select
its slot or add one in an empty cell. Edit local coordinates/capacity, remove
selected slots or use **Rename slot ID**. Inspection capacity remains 1 per slot.
Apply commits the whole table. Unused footprint cells may remain empty.

Expanding a slot facility creates slots only in the newly added area, preserving
old IDs, holes and bindings. Shrinking removes outside slots and their bindings
only after showing the affected references. Width and height in the property form
follow the same rules as the resize handles.

**Change storage mode** explicitly converts a Buffer. Pool to Slots shows a
capacity-1 draft that can be configured before Apply, with total capacity visible.
Slots to Pool asks for the total capacity or Unlimited. Incompatible bindings are
removed after review; slot access is never silently widened to the whole pool.
Pool and scrap capacity can be zero; slot capacities must be positive.

Select a Port and choose **Edit bindings**. Click exact slots or whole resources
on the map to add/remove targets; configure Pickup, Drop off, or both for each
applicable target. Chargers use Charge and scrap bins use Drop off. Pale yellow
fills preview targets, with no connecting lines. Nothing changes until Apply.

Changing selection, switching tabs, leaving Edit, saving or closing while a draft
is pending asks **Apply / Discard / Cancel**. Failed validation keeps the draft
available for correction. Slot/binding dialogs temporarily own map clicks.

### Undo and clipboard

Undo/Redo use an independent history per document. Saving marks a clean point
without clearing that history; undoing back to it removes the modified marker.

Copy/Cut/Paste use standard platform shortcuts: Command on macOS and Control on
Windows/Linux. Copy includes selected resources and their slots, rewires references
within that selection and clears references outside it, including same-factory
pastes. Paste previews at the cursor; click a legal position to apply once.
Escape cancels. Text fields retain their usual editing shortcuts.

## Save, templates and recovery

**Save** writes the complete v2 design; **Save As** selects a new destination for
this document. The first Save of a template copy asks whether to save a new file
(default) or update the source template. Updating a built-in template creates a
complete local override; it never modifies bundled original YAML. Future copies
use the override; already open documents stay unchanged. Subsequent Save uses the
chosen destination.

**File → Manage templates** creates a design from a listed template, registers
existing YAML files, restores a built-in original or removes a file-template
record. Removing a record keeps the actual file. **Save as Template** saves a
complete YAML and registers its path. The New dialog retains the two built-in
choices; additional registered templates are available in Manage templates.

Modified documents show a dot in their tab. Close offers Save/Discard/Cancel.
Invalid edits cannot be applied; incomplete valid designs with warnings can be
saved. External-file changes are detected using the last loaded/saved digest;
a conflict offers Save As, Reload and discard changes, or Cancel. Failed writes
retain the old file and the current undo history. YAML formatting is regenerated;
handwritten YAML comments are not retained.

Every 60 seconds, applied changes in modified documents receive a separate
recovery snapshot. Unapplied form or dialog drafts are not included. On startup,
recovery is offered before creating a new document; **File → Recover unsaved
designs** also opens the chooser. Restoring creates a new, modified Browse
document and leaves the source YAML untouched. Save or explicit close/discard
clears that document's snapshot.

Template records, full local overrides and snapshots live under Qt's writable
AppDataLocation (normally `~/Library/Application Support/SmartSOM/SmartSOM Studio`
on macOS), in `templates/` and `recovery/`. These are local application data, not
repository files or additional fields in the factory schema.

## Export

**File → Export map** previews a full-map PNG or SVG. Choose pixels per cell,
grid, numbers, ports, and no/selected/all binding highlights. The exported scene
omits current selection outlines, hover, resize handles and placement ghosts.
Exports do not change the document or its undo history. PNG is limited to 64
million pixels; use a smaller pixel size or SVG for larger maps.

Evaluation and pause/step/play/replay belong to the independent runtime window,
which shares Studio drawing components. They do not add execution to the editor.
Workload/scenario editing, smoke/stars animations and application bundles remain
outside this editor slice. Implementation does not authorize automatic commits.

## Checks

```sh
uv run --no-sync pytest -q tests/unit/test_factory_design.py tests/unit/test_factory_design_config.py tests/unit/test_studio_template.py tests/unit/test_studio_cli.py tests/unit/test_studio_editing.py
SMARTSOM_REQUIRE_STUDIO=1 uv run --no-sync pytest -q tests/integration/test_studio.py tests/integration/test_studio_editor.py
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Qt tests can use the offscreen platform; a visible macOS window is inspected
separately. Scale acceptance includes 100×100 cells, 100 machines, 200 buffers,
200 ports and 100 AGVs. Checks cover loading, selection, authoring transactions,
file round trips and view interaction. Checks establish
software behavior, not simulation support or research performance.
