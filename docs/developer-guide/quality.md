# Quality Checklist

UQGrid relies on a combination of automated checks and manual reviews to keep
the codebase healthy. Use this checklist as a reference when preparing changes.

## Automated tooling

| Tool | Command | Purpose |
| --- | --- | --- |
| Ruff | `python -m ruff check .` | Enforces style and catches common errors. |
| MyPy | `python -m mypy ...` | Type checking for critical modules. |
| Pytest | `pytest` | Functional and regression tests. |
| Vulture | `vulture uqgrid bin --min-confidence 80` | Dead-code identification (optional for large cleanups). |

These commands are aggregated in the `make lint`, `make test`, and `make ci`
targets for convenience.

## Profiling + performance

- Use `bin/dynamics_driver.py` with the `--profile` flag (if available) or
  Python profilers to spot hotspots.
- Capture representative input sizes before optimizing. Record the environment
  (CPU, interpreter, PETSc version).

## Documentation

- Update relevant MkDocs pages when introducing new modules, arguments, or
  workflows.
- For extensive architecture changes, add a short section under
  `docs/reference/`.
- Keep example commands copy-paste friendly and tested.

## Reviewing large diffs

Break large changes into reviewable commits. Aim for focused PRs:

- Device models vs. solver enhancements.
- Documentation-only updates.
- Performance/profiling adjustments.

## Release readiness

Before tagging a release candidate:

1. Ensure CI is green on the default branch.
2. Verify MkDocs builds without warnings (`Makefile` target `docs`).
3. Update version metadata in `pyproject.toml` and release notes.

Maintaining these habits keeps UQGrid reliable for research and production
studies alike.
