# SmartSOM Agent Instructions

These instructions apply to the entire repository. SmartSOM is being built as
a scientific simulator, so implementation claims and experiment evidence must
remain distinguishable.

## Required Context

Ordinary Markdown links are not loaded automatically. Use the triggers below
to load context, reusing already-read, unchanged documents within the same task:

- Before the first code, configuration, or durable-path change in a task, read
  `README.md`, `docs/architecture.md`, and `docs/roadmap.md`. For subsequent
  changes, read only newly relevant sections; refresh context when the source
  documents change or the task crosses into another contract.
- Before the first change in one of these areas, also read the relevant
  sections of its document:
  - learners, observations, encoders, rules, action permissions or training
    sampling: `docs/algorithm-components.md` and `docs/training-sampling.md`;
  - Scenario files, factory maps or Studio: `docs/scenario-authoring.md`;
  - Experiment YAML: `docs/configuration-guide.md`;
  - launch, batch, scheduler or run-control behavior:
    `docs/current-workflow.md` and `docs/unified-runtime-scheduler.md`;
  - metrics, evaluation protocols or comparison fairness:
    `docs/research-metrics.md` and `docs/evaluation-protocols.md`.
- Before creating or reorganizing commits, read `CONTRIBUTING.md` in full.
- Before changing a foundational decision, read the applicable record under
  `docs/decisions/` in full and add a superseding decision rather than silently
  rewriting history. `docs/decisions/README.md` indexes the records.

Write a new decision record only when a change alters paper-facing semantics
(reward, observation, metrics, termination, comparison fairness) or a public
contract that existing inputs, runs or checkpoints depend on. Record other
design choices in the pull request description.

For architectural changes, ask for direction before the affected implementation
if a foundational decision would change without authorization, or if a material
conflict remains unresolved after applying instruction precedence. Decide
implementation details within the authorized scope without asking again merely
because the documents do not specify them.

## Project Boundaries

- Describe as supported only what `docs/roadmap.md` lists under "Current
  implementation and acceptance". Do not describe planned capabilities as
  supported.
- Keep the simulator core independent of experiment orchestration, algorithm
  frameworks, external solvers, and generated artifact storage.
- Prefer composition and explicit typed contracts over deep inheritance or a
  monolithic environment class.
- Do not create placeholder packages for future roadmap items. Add a package
  only with its first real behavior and focused tests.
- Preserve semantic entity and action identities. Do not make simulator truth
  depend on transient array or candidate-slot positions.
- Treat event modules, resource/capability modules, and feasibility constraints
  as distinct concepts.
- Keep optional frameworks optional. Importing `smartsom` must not require any
  package that `pyproject.toml` lists only under an optional extra.
- Generated runs and artifacts are local evidence and are not committed unless
  a task explicitly authorizes a small fixture or example.

## Working Rules

- Inspect `git status --short --branch` before mutation and preserve unrelated
  user work.
- Development worktrees live under `artifacts/development/<topic>`. Each
  worktree carries its own `AGENTS.md` and `.agents/skills/`. When you start
  working in one, read that copy first; it can differ from the checkout the
  session started in.
- Use `uv` for Python versions, dependency management, and command execution.
- 优先使用所用工具或框架官方提供的命令、调度器和原生机制。官方方案
  无法满足需求，需要自定义或替换其 core、调度器等核心机制时，先向用户
  说明涉及的部分及具体原因（如功能不支持或已确认的限制）。通过项目
  YAML 配置和受支持扩展接口编写自定义 Python 属于正常扩展，无需额外提醒。
- Add no dependency until the code that uses it is part of the same change.
- Run focused tests first, then the repository quality gate appropriate to the
  change. Follow [CI ownership and impact rules](docs/ci.md) when adding tests or
  changing validation: new files need an explicit owner, native contracts must
  execute with their dependencies, and unknown impact expands to full verification.
- Keep source, tests, configuration, and documentation for one logical change
  in the same reviewable checkpoint.
- When authoring new formal `train` or `evaluate` experiments or batch
  directories, populate non-scientific `meta` (`short_title`, `long_title`,
  `short_description`, `full_description`) in the Experiment YAML or batch root
  `meta.yaml` (`schema: smartsom.experiment-meta/v1`) so terminal monitors,
  W&B, and external widgets share one source of truth.
- A passing test establishes engineering behavior only; it does not authorize
  a research claim, milestone promotion, release, or formal experiment.
- Formal experiments must run from an integrated `main` commit or an explicit
  tag and must record that source identity in their manifest.
- When proposing or changing a scenario or simulator design setting (layout,
  aisle width, buffer and port arrangement, fleet size, mover behavior), name
  a precedent from literature published in 2016 or later, or from real factory
  practice, and say whether you read the source or only a summary of it. Flag
  a setting without such a precedent as open instead of presenting it as
  settled.

## Repository skills

Skills live in `.agents/skills/`; `.claude/skills` links to the same directory.

- For evaluation timing distributions and optimization validation, use
  [evaluation-profile](.agents/skills/evaluation-profile/SKILL.md).
- After a reviewable code, test or runtime-configuration change, or for a requested
  code review, use [code-review](.agents/skills/code-review/SKILL.md). Select review
  depth and reviewer count by risk; do not default to a full multi-agent review.
  Pure prose uses lightweight checks. Performance measurements apply when the
  affected behavior warrants them. Respect the user's scope and active delegation
  limits; the skill does not authorize publishing, merging or formal experiments.
