# Contributing to Retrieve-Me

Thanks for helping build Retrieve-Me. This document is the whole workflow:
setup, daily development, code standards, and the pre-merge checklist. It exists so
that review is about the change, not about style.

## Setup

```bash
git clone git@github.com:abj360/Retrieve-Me.git
cd Retrieve-Me
pip install -r requirements.txt
pre-commit install
cp .env.example .env
```

Run the full stack (api, dashboard, Qdrant, Redis) with:

```bash
scripts/run_local.sh            # or: docker compose -f docker/docker-compose.yml up --build
```

## Daily workflow

- **Every feature lands on a feature branch**, never directly on `main`.
  Branch naming: `<type>/<short-description>`, e.g. `feat/rrf-fusion`,
  `fix/qdrant-pool`. Push your branch to **your own fork** and open the PR against
  `abj360/Retrieve-Me`. Never push branches from someone else's account.
- **Keep PRs small enough to review in one sitting.** Split large features into a
  stack of small PRs rather than one giant one.
- **Commit style:** Conventional prefixes — `feat`, `fix`, `perf`, `refactor`,
  `test`, `chore`, `docs`, `ci`, `style` — followed by a plain-language summary of
  the change (not a restatement of the diff). Commits are small and atomic: one
  logical thing per commit, revertible on its own.
- **Commit as yourself.** `git config user.name` / `user.email` must be your own
  GitHub identity before you commit. No tool identities, no co-author trailers.

## Code standards

The full rules live in the team's engineering-standards doc; the enforced
highlights:

- **File headers (Python):** shebang, then a structured module docstring —
  `filename.py --- role`, a blank line, and a `Contains:` block listing every
  function/class the file exposes. Imports follow: stdlib, blank line, third-party,
  blank line, local — alphabetized within each group.
- **File headers (TS/JS):** same convention — `#!/usr/bin/env ts-node` (or `node`),
  then a `/** ... */` block with the same `Contains:` list. Bundlers strip the
  shebang, so it is safe in browser-target files.
- **Docstrings on everything.** Every function, method, and class — no exceptions.
  Line 1 starts with a third-person verb ending in "s" (`Creates`, `Validates`,
  `Resolves`). `Args:` / `Returns:` sections only when there is something to say;
  `Attributes:` on classes that hold state. If a section needs more than a couple of
  lines, the function is doing too much — split it.
- **Comments are the exception.** Code explains itself through naming and
  structure. Comment only the genuinely non-obvious: a business rule, a library
  workaround, a deliberate deviation, a magic constant with a real source. No
  commented-out dead code — git history is the archive.
- **Formatting:** Ruff for Python (lint + format, line length 100, double quotes),
  ESLint + Prettier for TS/JS, Prettier for YAML/JSON/Markdown. Zero warnings on
  merge, locally and in CI.
- **Types:** full hints on every Python signature (`mypy --strict`), `strict: true`
  in TS. No `Any`/`any` without a one-line justification.
- **Structure:** one responsibility per function; functions stay under ~30 lines;
  guard clauses over pyramids; no magic numbers — name constants; no side effects at
  import time; composition over inheritance; dependency injection over globals;
  frozen dataclasses / readonly types where reasonable.
- **Errors:** catch specific exceptions. Fail closed, not open — a timeout or an
  unexpected state blocks/rejects, never silently passes. Errors that matter get
  logged with enough context to debug without reproducing.
- **Secrets:** none in code, ever. `.env` is gitignored; `.env.example` holds
  non-secret defaults.

## Testing policy

- New logic ships with tests **in the same PR**, not a follow-up ticket.
- Tests live next to what they test (`tests/unit`, `tests/integration`,
  `tests/smoke` mirroring `src/`).
- A test asserts on real behavior. A mocked dependency **must be able to fail** in
  the test — if the mock always succeeds, the test proves nothing. (This rule exists
  because of the 2026-05 smoke-test incident; see the git history around
  2026-05-12.)

## CI gates (all must pass locally first)

- `ruff check .` and `ruff format --check .` (ESLint + Prettier for the dashboard)
- `mypy src` (`tsc --noEmit` for the dashboard)
- `pytest -q` (unit, integration, smoke; CI also boots Qdrant + Redis services)
- `pip-audit` dependency vulnerability scan
- `docker compose -f docker/docker-compose.yml up --build` boots the whole stack
  cleanly

## Pre-merge checklist

- [ ] Ruff / ESLint clean, formatter applied
- [ ] `mypy --strict` / `tsc --strict` clean
- [ ] Every new function/class has a verb-first docstring, with
      Args/Returns/Attributes filled in where relevant
- [ ] No commented-out code, no restating-the-obvious comments
- [ ] Tests added alongside the change, and they can actually fail
- [ ] Commit authored and committed as yourself — your identity, no tool trailer
- [ ] `docker compose up --build` still boots the whole stack cleanly

## Review

Peter (the repo owner) has final review on all merges. Expect review comments on
design before style: if a design doc or PR description says "this will scale,"
expect to be asked "to what number, measured how." ADRs (`docs/adr/`) come before
code on anything architecturally load-bearing.
