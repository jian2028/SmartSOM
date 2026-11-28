# ADR 0085: Unified run metadata, display selection and read-only monitor summary

- Status: Proposed for PR #51; engineering verification recorded separately.
- Supersedes / extends: Extends [ADR 0058](0058-current-runtime-and-tracking.md) and [ADR 0078](0078-unified-operational-cli.md) without altering simulator physics, rewards, observations, checkpoints, or scientific identity hashes.

## Context

Operators and external desktop monitors (such as Übersicht panels watching local Mac and remote server runs) previously had to maintain separate sidecar maps for short batch codenames and human descriptions, while terminal monitors and W&B runs derived names from directory paths. Each frontend parsed different files and lacked a single contract for titles, descriptions, compiled run facts, and a future unified ETA payload.

## Decision

1. **Non-scientific `meta` contract (`smartsom.experiment-meta/v1`)**:
   - Single experiments accept an optional `meta` block (`short_title`, `long_title`, `short_description`, `full_description`, `tags`).
   - Directory batches accept an optional root `meta.yaml` with `schema: smartsom.experiment-meta/v1` (ignored by `entrypoints()`).
   - Launch CLI accepts `--short-title`, `--long-title`, `--short-description`, `--full-description`, `--meta-tag`, and `--allow-empty-meta`.
   - Priority is `batch meta.yaml` < `experiment.yaml meta` < CLI flags.
   - `meta` is excluded from `scientific_documents()` and forbidden as a `matrix` axis: changing titles or descriptions never changes `scientific_sha256` or checkpoint compatibility.
   - Missing authored fields stay `null` in JSON (never `"N/A"`). Resolved payloads expose both authored fields, a deterministic `auto_summary` derived from compiled `facts`, and precomputed `effective` fallbacks.

2. **Launch guardrails without external API dependencies**:
   - New formal CLI launches (`train` / `evaluate`, excluding activations of already-frozen incremental batches) require `short_title` and `short_description` unless `--allow-empty-meta` is supplied.
   - Interactive TTY runs prompt for missing required fields after printing compiled `facts`. Non-interactive CLI runs fail with an explicit configuration error naming the missing fields and flags.
   - `smoke` and `calibrate` runs do not require `meta`.

3. **Compiled `facts` and reserved `eta` contract**:
   - Compiled `facts` (`host`, `created_at`, `task`/`tasks`, `task_count`, `seeds`, `budget`, `algorithms`, `observations`, `controls`, `scenarios`, `factory_ids`, `scales`, `machines`, `agvs`, `orders`, `concurrency`, `tracking`) are derived automatically and frozen into `plan.json` / `run.json` (and child `manifest.json`), then surfaced in `logs/progress.json` and `catalog.describe`.
   - `logs/progress.json` and `smartsom runs --summary --json` reserve a uniform `eta` object (`schema: smartsom.eta/v1`, `remaining_seconds`, `eta_at_unix`, `confidence`, `basis`, `breakdown`, `updated_at`), initially `unavailable` until the follow-up scheduler ETA PR populates it.

4. **Configurable per-consumer display mapping**:
   - `display.title_field` (`short_title | long_title | combined | default`) and `display.description_field` (`short_description | full_description | none`) control terminal monitor rendering.
   - `tracking.name_field` (`short_title | long_title | default`) and `tracking.notes_field` (`full_description | short_description | none`) control W&B `name` prefixing and `notes` when W&B is enabled. `tracking_group` uses `short_title` as its prefix when present; `tags` add `title:<short_title>` and `host:<hostname>`; `config` records full `meta` and `facts`.
   - `smartsom runs --summary --json` (`schema: smartsom.monitor-summary/v1`) provides one read-only top-level host + batch summary for external monitors without duplicating nested child trial directories.

## Consequences

- Existing runs without `meta` or `facts` remain readable and compatible.
- External monitors can select whichever title/description fields fit their layout from one JSON endpoint without maintaining hardcoded sidecar maps.
- Follow-up PR (scheduler-embedded ETA engine) fills `eta` in place without breaking consumers.
