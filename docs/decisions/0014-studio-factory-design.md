# 0014 — Studio factory design and read-only preview

Date: 2026-09-11

Status: Accepted

## Decision

SmartSOM Studio is an optional local desktop application built with PySide6,
Qt Widgets and Graphics View. Its first slice provides immutable factory design
data, validated YAML, Template 1 and a read-only preview. Editing, evaluation,
playback, animation and desktop application packaging are separate later slices.
Each implementation stage stops for user review; no automatic commits are made.

`FactoryDesign` and `smartsom.factory/v2` are authoring contracts. They do not
replace executable `FactorySpec` or the existing v1 factory resolver. Studio
does not import or migrate v1 files, and v1 execution does not silently ignore
unsupported v2 fields. Existing configurations, experiments and their evidence
remain unchanged.

This adds an explicit authoring representation to ADR 0002. It does not change
its workload/scenario/algorithm/run ownership. In particular, neither a layout
nor its preview owns workloads, simulation clocks, random draws or metrics.
The fixed-matrix logistics of ADR 0006 and final-only inspection of ADR 0008
remain the runtime contracts. Physical inspection stations, scrap sinks and
energy declarations in the new design are not claims of runtime support.

## Ownership and identities

`domain.factory_design` owns standard-library immutable data and geometry.
`config.factory_design` owns strict parsing, diagnostics and reliable file
saving using the existing Pydantic/PyYAML stack. Neither imports Qt.
`studio` owns window lifecycle, viewing and human interactions. Qt is imported
only when that optional interface is launched. The CLI dispatch is thin and
does not enter the experiment runner.

All design information is contained in the same versioned YAML, including when
that file is used as a template. Templates are copied documents, not inherited
configuration layers. Personal UI settings may retain file paths and view
preferences, but cannot contain additional factory truth.

Resources have globally unique semantic IDs within a factory; slots have
owner-local IDs and quality modes have machine-local IDs. IDs are case-sensitive
ASCII letters followed by letters, digits, underscores or hyphens. Generated
IDs use names such as `machine_001`. Unicode display names are independent.

## Spatial contract

Coordinates are abstract integer grid cells, with origin at the top left.
Footprint width/height and slot-local coordinates describe the **current**
orientation. Rotation metadata must not rotate these coordinates a second time.
The planned rotation command preserves the selection bounding box's top-left
corner, swaps dimensions and transforms slot coordinates clockwise.

A port is exactly one walkable cell, with a set of allowed headings and typed
target/operation bindings. It contains no child docks. Two ports cannot share
a cell. Multiple ports can reference the same target; a port can reference
multiple targets. Explicit binding defines access; no adjacency or range rule
is inferred. Ports do not follow a separately moved resource automatically.

All footprint-bearing resources, including chargers, are solid. AGVs may occupy
port cells but not solids or another AGV's initial cell. Slots are internal
positions of their owner, not separate obstacles. Machine pre/post buffers are
independently optional, with ownership stored only on the buffer. A machine's
single-job processing capacity is independent of its footprint.

## Validation and evidence

Invalid types/ranges, IDs, references or geometry are errors. Incomplete but
structurally valid drafts can be saved, with warnings for unbound ports,
unassigned machine buffers or empty slot facilities. An empty factory and a
machine without buffers are valid designs.

Saving checks the expected byte digest and publishes a complete temporary file.
Create-only saving cannot replace a concurrent create. Replacement uses an
optimistic conflict check, not a lock protocol with third-party editors. YAML
formatting is regenerated, including short system comments; handwritten comments
are not retained.

Static geometry tests, file round trips and GUI checks establish editor behavior
only. They do not establish traffic feasibility, collision avoidance, deadlock
freedom, simulation equivalence or research performance.

See [factory design](../factory-design.md) for the complete field and template
contract and [Studio](../studio.md) for the implemented interface and deferred
editing workflow.
