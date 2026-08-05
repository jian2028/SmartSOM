# 0001 — Project Foundation

Date: 2026-08-05

Status: Accepted

## Context

SmartSOM is starting as a new repository intended to grow into a modular
dynamic FJSP simulator and experiment platform. A previous SmartSOM generation
provides useful lessons, but this repository should not inherit legacy code or
complex governance before concrete requirements justify it.

## Decision

- Use the `smartsom` package name and English-only repository content.
- Use Python 3.12, `uv`, a committed lockfile, Hatchling, pytest, and Ruff.
- Begin with an installable scaffold and a static FJSP vertical slice.
- Keep a thin semantic core and compose future behavior through explicit
  module and adapter contracts.
- Keep external solvers, learning frameworks, MARL frameworks, and experiment
  tracking optional.
- Use stable entity-based semantic actions rather than candidate-slot action
  identity.
- Use the same single-run execution core for manual and batch execution.
- Treat local manifests, semantic traces, metrics, summaries, and retained
  failures as the experiment authority; add W&B only as an optional mirror.
- Use concise `AGENTS.md` routing and `CONTRIBUTING.md` workflow rules without
  initializing a separate memory bank.
- Use short-lived branches and Conventional Commits. Squash one-objective pull
  requests by default; preserve commits with rebase merge only when every
  commit is deliberately curated and independently valuable.

## Consequences

The initial repository contains no scheduling or simulation behavior and makes
no support claims. Future packages are created only with their first tested
behavior. Formal experiments must identify an integrated `main` commit or tag,
and optional framework integration cannot become a base-package dependency.
