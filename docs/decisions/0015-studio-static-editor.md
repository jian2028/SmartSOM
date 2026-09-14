# 0015 — Complete static editor and document lifecycle

Date: 2026-09-13

Status: Accepted

## Decision and scope

Implement the agreed static editor as one delivery for user review, without
creating commits automatically. This supersedes only ADR 0014's staged delivery
schedule for editing, recovery/templates and image export. Its format, spatial,
identity and simulator separation decisions remain in force.

Every opened file, template copy, blank design and recovered document starts in
Browse. Explicit Edit enables mutations. Typed parameter forms and slot/binding
dialogs keep unapplied drafts; Apply creates one validated transaction. Switching
selection, documents or mode, saving and closing resolve drafts with
Apply/Discard/Cancel. Applied edits have independent per-document undo histories.
Saving marks a clean point without removing history.

Solid resources start at one cell: click places 1×1, and dragging defines one
rectangle. AGVs and ports are fixed one-cell resources. Successful placement
returns to Select unless Continuous placement is enabled. Invalid candidates
are rejected atomically. Only explicitly selected resources move or rotate.
Slot expansion preserves holes/IDs and adds slots only in newly added area.
Structural removals show affected resources, slots and bindings before Apply.

## Files and templates

Each design remains one complete `smartsom.factory/v2` YAML. A template's first
save chooses a new file by default or explicitly updates its source. Updating a
built-in template writes a complete local YAML override; bundled originals stay
available through Restore original. Overrides are replacement documents, not
configuration inheritance. File templates are ordinary YAML paths in a local
catalog; removing a catalog entry never deletes the user's YAML.

Save As changes the document destination. Save uses the previous byte digest to
detect external changes and preserves the old file on failure. A source already
open in another tab cannot be overwritten through a different tab.

Modified, applied documents receive separate recovery snapshots every 60 seconds.
Unapplied property/slot/binding drafts are not recovered. Recovery creates a new
unsaved document, not a replacement of its original source. Normal save or an
explicit close/discard clears that document's recovery snapshot.

PNG/SVG export uses a separate scene with configurable grid, numbers, ports and
binding highlights. It covers the full map and excludes selection outlines,
resize handles, hover and transient placement previews.

## Evidence boundary

Static editor tests and UI checks do not demonstrate simulator capabilities.
Workload/scenario editing, integer-tick runtime changes, evaluation, replay,
animation and application packaging still require separate work.
