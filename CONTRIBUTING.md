# Contributing to SmartSOM

## Development Setup

Use the repository-pinned Python version and locked environment:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Do not install optional solver, learning, multi-agent, or tracking dependencies
until an approved change introduces the corresponding adapter.

## Branches

After the initial repository scaffold, keep `main` usable and work on one
short-lived branch per logical objective. Use readable kebab-case names with
one of these prefixes:

```text
feat/<topic>
fix/<topic>
docs/<topic>
test/<topic>
refactor/<topic>
perf/<topic>
chore/<topic>
experiment/<topic>
```

Do not create a long-lived `develop` branch. Experiment identity belongs in
configuration, run IDs, manifests, and Git source identities rather than in
permanent experiment branches.

## Commits

Use Conventional Commits:

```text
type(scope): imperative summary
```

Initial scopes are `repo`, `domain`, `engine`, `dispatch`, `modules`,
`algorithms`, `experiments`, `trace`, `telemetry`, `docs`, and `ci`.

Examples:

```text
chore(repo): initialize project scaffold
feat(domain): define static FJSP specifications
feat(engine): implement deterministic event calendar
fix(dispatch): reject infeasible machine assignments
test(trace): verify deterministic replay
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

The first scaffold commit is the sole bootstrap exception and may establish
`main` directly because no base commit exists yet.

## Experiment Source Identity

Formal experiments must start from an integrated `main` commit or an explicit
tag. Development smoke runs may use feature branches, but their commit IDs must
not be cited as durable evidence when the branch will be squashed or rewritten.

Generated `runs/` and `artifacts/` content remains uncommitted unless a small,
intentional fixture or example is explicitly approved.
