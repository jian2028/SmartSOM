# Contributing to SmartSOM

## Development Setup

Use the repository-pinned Python version and locked environment:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

`uv sync --locked` installs the minimal environment used for lint and the
core tests. Learning, tuning, tracking and Studio paths need their extras from
`pyproject.toml`; the README shows the usual research environment
(`uv sync --locked --extra learning --extra cpu --extra wandb`). Add no new
dependency or extra until the code that uses it is part of the same change.

See [tiered CI and test ownership](docs/ci.md) for automatic impact selection,
full verification, local reproduction and requirements for new tests.

## Branches

Keep `main` usable and work on one short-lived branch per logical objective.
Use readable kebab-case names with one of these prefixes:

```text
feat/<topic>
fix/<topic>
docs/<topic>
test/<topic>
refactor/<topic>
perf/<topic>
chore/<topic>
ci/<topic>
experiment/<topic>
codex/<topic>
```

`codex/<topic>` is for branches an agent session creates. The pull request
title still carries the Conventional Commit type.

Do not create a long-lived `develop` branch. Experiment identity belongs in
configuration, run IDs, manifests, and Git source identities rather than in
permanent experiment branches.

## Commits

Use Conventional Commits:

```text
type(scope): imperative summary
```

Scopes in use are `repo`, `domain`, `engine`, `algorithms`, `learning`,
`experiments`, `evaluation`, `telemetry`, `studio`, `config`, `cli`, `docs`,
and `ci`. Prefer one of these; add a scope only for a new top-level area.

Examples:

```text
feat(engine): implement deterministic event calendar
fix(experiments): keep paused trials out of the admission queue
perf(algorithms): reuse shared critic features within a frame
test(telemetry): verify recorder isolation
docs(architecture): clarify module boundaries
```

Each commit should be independently reviewable and revertible. Keep directly
supporting tests and required documentation with the behavior they verify.

## Pull Requests and Merging

- A pull request should contain one logical objective.
- The pull request title must be the intended Conventional Commit message.
- Require the CI quality gate before merging.
- Use squash merge by default so one reviewed objective becomes one `main`
  commit.
- Use rebase merge only when every branch commit has been deliberately curated,
  independently tested, and is worth preserving on `main`.
- Split unrelated objectives into separate branches and pull requests rather
  than combining them in one squash.

## Pull Request Reviews

These conventions apply to any review posted on a GitHub pull request, whether
written by a person, the `@claude` workflow or another agent.

- Write GitHub reviews in English, whatever language was used to request them.
- Open one inline thread per independently fixable finding that maps to a
  changed line, and submit the threads together in one review when possible.
- Keep the summary comment for the overall assessment and cross-cutting
  feedback; do not repeat the inline findings there.
- After fixing a finding, reply in its thread with `Fixed in <sha>` and the
  relevant verification. Resolve the thread once the fix has been checked.

## Decision Records

`docs/decisions/README.md` indexes the records and says when one is needed.
Give a new record the next number after the highest one on `main` at merge
time, and renumber on the branch if another record landed first. Add its row
to the index in the same change.

## Experiment Source Identity

Formal experiments must start from an integrated `main` commit or an explicit
tag. Development smoke runs may use feature branches, but their commit IDs must
not be cited as durable evidence when the branch will be squashed or rewritten.

Generated `runs/` and `artifacts/` content remains uncommitted unless a small,
intentional fixture or example is explicitly approved.
