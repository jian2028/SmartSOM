# Monitor Summary Export Contract (`smartsom.monitor-summary/v1`)

This document defines the read-only JSON contract shared by terminal tools and external desktop monitors (such as Übersicht widgets watching Mac and remote Linux servers).

## Command

```sh
# Local machine
uv run --no-sync smartsom runs --summary --json

# Custom roots / active only / custom recent history cap
uv run --no-sync smartsom runs --summary --json --root runs --recent 15
uv run --no-sync smartsom runs --summary --json --active-only
```

- Purely read-only; creates no files and submits no scheduler commands.
- Nested child trial directories (`trials/.../seed-*`) owned by a parent batch are automatically folded into their parent batch so each batch appears once.

## Top-Level Payload

```json
{
  "schema": "smartsom.monitor-summary/v1",
  "generated_at": 1795774500.12,
  "host": {
    "name": "server-01",
    "platform": "Linux",
    "machine": "x86_64",
    "cpu_percent": 84.0,
    "cpus_used": 26.88,
    "cpus_total": 32,
    "memory_used_bytes": 34359738368,
    "memory_total_bytes": 134217728000,
    "memory_used_gib": 32.0,
    "memory_total_gib": 125.0,
    "memory_percent": 25.6
  },
  "counts": {
    "running": 6,
    "queued": 1,
    "paused": 0,
    "completed": 12,
    "failed": 0,
    "total": 19
  },
  "batches": [ ... ]
}
```

## Per-Batch / Run Entry

Each element of `batches` has the following stable structure (active first, then queued/paused, then recent terminal sorted by `updated_at` descending):

```json
{
  "id": "85e1d843d7024f1b82cb607dafc42098",
  "short_id": "a1b2c3d4e5f6",
  "directory": "/path/to/runs/a5",
  "kind": "batch-directory",
  "status": "running",
  "stage": "stage_0",
  "meta": {
    "short_title": "a5",
    "long_title": "K4 五车全角色学习对比",
    "short_description": "K4 五车，全角色学习",
    "full_description": "多行完整实验背景与设计动机…",
    "tags": ["k4", "five-agv"],
    "auto_summary": "small · dqn/base · 1000ep · s101-104",
    "effective": {
      "short_title": "a5",
      "long_title": "K4 五车全角色学习对比",
      "short_description": "K4 五车，全角色学习",
      "full_description": "多行完整实验背景与设计动机…"
    }
  },
  "facts": {
    "host": { "name": "server-01", "platform": "Linux", "machine": "x86_64" },
    "created_at": 1795770000.0,
    "input_type": "batch",
    "task": "train",
    "tasks": ["train"],
    "task_count": 4,
    "selected_task_count": 4,
    "seeds": { "selected": [101, 102, 103, 104], "all": [101, 102, 103, 104] },
    "budget": {
      "train_episodes": 1000,
      "train_ticks": null,
      "periodic_evaluation_rounds": 20,
      "periodic_evaluation_cases": 12,
      "final_evaluation": true,
      "test_cases": 80
    },
    "algorithms": ["dqn"],
    "observations": ["base"],
    "controls": ["distributed"],
    "scenarios": ["small.yaml"],
    "factory_ids": ["small_factory"],
    "scales": ["small"],
    "machines": [8],
    "agvs": [10],
    "orders": [300]
  },
  "progress": {
    "active_tasks": 2,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "paused_tasks": 0,
    "queued_tasks": 2,
    "deferred_tasks": 0,
    "removed_tasks": 0,
    "total_tasks": 4,
    "fraction": 0.09,
    "episodes": { "min": 178, "max": 182, "budget": 1000 },
    "physical_ticks": { "completed": 182000, "budget": null }
  },
  "eta": {
    "schema": "smartsom.eta/v1",
    "remaining_seconds": null,
    "eta_at_unix": null,
    "confidence": "unavailable",
    "basis": "eta engine pending",
    "breakdown": null,
    "updated_at": 1795774498.0
  },
  "wandb_url": "https://wandb.ai/...",
  "updated_at": 1795774498.0
}
```

## Consumer Field Selection Guide (e.g. Übersicht)

- **Compact card row**: read `meta.effective.short_title` (badge) and `meta.effective.short_description` (one-line description).
- **Expanded detail popover**: read `meta.effective.long_title`, `meta.effective.full_description`, `facts.seeds`, `facts.algorithms`, `wandb_url`.
- **Authored vs fallback**: raw `meta.short_title` / `meta.short_description` / `meta.long_title` / `meta.full_description` are `null` when not authored (never `"N/A"`). Use `meta.effective.*` for guaranteed display text, or check raw fields when you want to distinguish authored text from fallback text.
- For an incremental batch, `facts.task_count` and `facts.seeds.all` cover its committed inventory. The selected count and seeds reflect the latest submission plus same-batch tasks accepted by `add` during that submission. The original launch plan stays frozen.
- Episode counts and case fractions use committed work. A child can finish its wall-clock allocation before reaching its episode budget.
- **ETA readiness**: display ETA when `eta.remaining_seconds !== null` (and optionally count down locally using `eta.eta_at_unix`). When `eta.confidence === "unavailable"`, hide or show a neutral placeholder until the scheduler ETA PR populates it.
