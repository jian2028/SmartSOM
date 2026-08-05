# SmartSOM Agent Instructions

These instructions apply to the entire repository. SmartSOM is being built as
a scientific simulator, so implementation claims and experiment evidence must
remain distinguishable.

## Required Context

Ordinary Markdown links are not loaded automatically. Read each triggered
document completely before making the corresponding change:

- Before changing code, configuration, or durable paths, read `README.md`,
  `docs/architecture.md`, and `docs/roadmap.md`.
- Before creating or reorganizing commits, read `CONTRIBUTING.md`.
- Before changing a foundational decision, read the applicable record under
  `docs/decisions/` and add a superseding decision rather than silently
  rewriting history.

If these documents conflict or do not cover a requested architectural change,
stop and ask for direction before implementation.

## Project Boundaries

- The current repository is a scaffold until a roadmap milestone is
  implemented and verified. Do not describe planned capabilities as supported.
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
- Keep optional frameworks optional. Importing `smartsom` must not require
  Gymnasium, Ray, PettingZoo, Torch, OR-Tools, or W&B.
- Generated runs and artifacts are local evidence and are not committed unless
  a task explicitly authorizes a small fixture or example.

## Working Rules

- Inspect `git status --short --branch` before mutation and preserve unrelated
  user work.
- Use `uv` for Python versions, dependency management, and command execution.
- Add no dependency until the code that uses it is part of the same change.
- Run focused tests first, then the repository quality gate appropriate to the
  change.
- Keep source, tests, configuration, and documentation for one logical change
  in the same reviewable checkpoint.
- A passing test establishes engineering behavior only; it does not authorize
  a research claim, milestone promotion, release, or formal experiment.
- Formal experiments must run from an integrated `main` commit or an explicit
  tag and must record that source identity in their manifest.
