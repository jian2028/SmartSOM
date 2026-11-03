# 0066 — Bind execution subsets to ordered Experiment selections

Status: accepted; implementation independently reviewed and locally validated.
Publication and deployment remain separate actions.

Supersedes ADR0065's unpublished `--order` interface. Its immutable inventory,
append transactions, stage guards and source protections are unchanged.

Repeated `--id` both selects and defines eligible admission priority. An optional
`--select` YAML mapping binds to the preceding ID. Only `train_seeds` is allowed;
its values reuse the Experiment schema (list of actual seeds or count beginning
at 101). Selectors must be subsets of the full authored seed list. Repeated IDs
are allowed only when their expanded task sets do not overlap. Whole-selection
validation precedes allocation or queue publication. No selector changes science.

First-N remains an independent global execution selector; mixing it with local
selectors or scientific seed overrides is rejected. Unselected tasks retain
`deferred` status. Default full-directory ordering remains lexical discovery then
authored seeds. Stage dependencies and runtime priority/holds take precedence;
multiple workers do not promise completion order. Named Experiment directories
freeze complete inventories even when the first submission has no explicit seed
selector, enabling later supplement without changing output identities.

`--resume-missing` selects from the complete current inventory, skips finished
work, continues pending final evaluation and retries failed/interrupted training
only through intact compatible continuation data. It checks continuation file
hashes before publication; the native learner performs its existing format and
state restoration checks before updates. A missing/corrupt continuation fails
closed. No fresh learner is substituted. Manual task holds/removal require
explicit controls. A live owner rejects a second coordinator. Historical source
identity is not weakened by this selection feature.

These contracts are exercised with bounded fixtures and short engineering runs,
not new formal scientific experiments.
