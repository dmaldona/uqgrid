# Contributing Guide

Thank you for considering a contribution to UQGrid! This guide summarizes the
workflow for proposing changes. Refer to the root `CONTRIBUTING.md` for the
full policy.

## Development environment

1. Fork the repository and clone your fork.
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-enhancement
   ```
3. Install dependencies with development extras:
   ```bash
   pip install -e ".[dev]"
   ```
4. (Optional) Install PETSc if your changes exercise adjoint solvers.

## Coding standards

- Format and lint with `make lint` (ruff + mypy).
- Maintain Python 3.8+ compatibility; prefer modern typing but avoid
  `match` / pattern matching.
- Keep functions small and testable. Consider docstrings for public APIs.
- Document new configuration options in `docs/` and update the changelog or
  release notes when relevant.

## Tests

Run the full suite locally before opening a pull request:

```bash
make test
```

Skip PETSc-heavy adjoint tests with:

```bash
make test-fast
```

When adding functionality, accompany it with unit tests in `tests/`. Use pytest
fixtures for sample systems when possible.

## Pull request checklist

- [ ] All tests pass locally.
- [ ] Linting succeeds (`make lint`).
- [ ] Documentation updated (MkDocs pages + README when applicable).
- [ ] CI badges in the README remain valid.
- [ ] Large datasets or generated artifacts are excluded via `.gitignore`.

## Communication

- Open an issue to discuss significant changes before implementation.
- Tag maintainers when a pull request is ready for review.
- Provide benchmarks or profiling data for performance-sensitive features.

We appreciate your help improving UQGrid! 🎉
