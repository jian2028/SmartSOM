# 0016 — Factory operation catalog and portable authoring preferences

Date: 2026-09-14

Status: Accepted for the static Studio configuration slice.

## Decision

Extend the v2 factory with one authoritative operation-type catalog. Machines
reference its semantic IDs; unused types and shared/multiple capabilities are
valid. Unknown references are errors; an empty machine capability is a draft
warning. New machines receive corresponding standard capabilities in automatic
mode. Manual additions/deletions stop automatic growth. Deleting a referenced
type requires changing its machine references first; deleting machines does not
renumber or remove types. Exact editing and compatibility behavior is specified
in [factory design](../factory-design.md).

Store `authoring.operation_catalog_mode` alongside `factory` in the same versioned
file. This refines ADR 0014/0015's complete-document envelope: factory facts remain
in `factory`, portable editing preferences live in `authoring`, and view/layout
preferences remain local. No second factory truth, embedded workflow or simulator
state is introduced. Document transactions include both factory and authoring
snapshots; full-file I/O is mandatory for Studio save/template/recovery paths.

Old v2 files without a catalog are normalized in memory using the standard range
and already referenced types, without assigning missing machine capabilities or
writing the original. Explicit catalogs remain strict. Bundled templates are
updated; existing user files and local overrides are not rewritten in bulk.

## Boundary and next work

Studio edits Factory only. Runtime v1, workloads, scenario modules, algorithm
interfaces and existing evidence remain unchanged. This delivery does not add
placeholder Buffer agents or dynamic execution. A later separate branch must
specify and implement capability-based workload matching, AGV grid actions,
Buffer job-scoring/selection interfaces and timing, inspection/scrap/energy behavior,
and corresponding replay before retiring v1 execution. The core owns simulation
logic; Studio owns editing, so training never requires an open desktop editor.

Capacity behavior is retained: new slot buffers have one capacity-one slot per
cell; slot capacities are editable; explicit finite/unlimited pools remain.
An absent facility does not inherit the old runtime's infinite-buffer convention.

## Verification

Cover catalog/default/manual transitions, reference deletion guards, cancellation,
undo/redo, full-file round trips, read-only legacy normalization, clipboard import,
template/recovery persistence and unchanged capacity behavior. Run focused Studio
checks and the repository quality gate, with explicit optional-backend limitations.
A passing static editor check does not establish dynamic simulation support.
