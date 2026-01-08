# Repository Guidelines

## Project Structure & Module Organization
- `uqgrid/`: core Python package (simulation, IO parsing, models, sensitivities).
- `tests/`: pytest suite, including adjoint tests marked with `@pytest.mark.adjoint`.
- `bin/`: runnable scripts (e.g., `dynamics_driver.py`, `generate_scenarios.py`).
- `data/`, `simulation_data/`: sample input datasets and artifacts.
- `docs/`: MkDocs documentation sources.

## Architecture Overview
- `uqgrid/core/`: base model types and shared system data structures.
- `uqgrid/io/`: loaders and parsers for PSS/E-style input files.
- `uqgrid/models/`: generator, exciter, governor, and load model definitions.
- `uqgrid/simulation/`: DAE integration, dynamics orchestration, and sensitivity/gradient workflows.
- `uqgrid/utils/`: small helpers (partitioning, tooling utilities).

## Build, Test, and Development Commands
- `make install`: install the package in editable mode.
- `make install-dev`: install dev dependencies (pytest, ruff, mypy, etc.).
- `make install-petsc`: enable PETSc support for adjoint sensitivities.
- `make test`: run the full pytest suite.
- `make test-fast`: skip adjoint tests (`-m "not adjoint"`).
- `make lint`: run `ruff` and `mypy` checks.
- `make docs-serve`: serve documentation locally with live reload.

## Coding Style & Naming Conventions
- Python 3.8+; prefer 4-space indentation and standard PEP 8 naming.
- Use `snake_case` for functions/variables, `CapWords` for classes, and `UPPER_SNAKE_CASE` for constants.
- `ruff` is the primary linter (line length 100 in `pyproject.toml`); `mypy` is used on selected modules.
- `black`, `isort`, and `pre-commit` are available via dev dependencies; run `pre-commit install` after setup.
- Performance-sensitive routines (residuals/Jacobians) should avoid in-function imports and runtime column sorting (no `argsort` in these paths). Ensure CSR column indices are ordered up front in preallocation and keep the same order in `residual_jac`.

## Testing Guidelines
- Framework: `pytest` with markers for slow adjoint validation tests.
- Name tests as `test_*.py` and functions as `test_*`.
- Use `make test-fast` during local iteration; use `make test` before opening a PR.
- For Jacobian verification, use the built-in finite-difference checks in `IntegrationConfig` (`check_jacobian`, `jacobian_check_tol`, `jacobian_check_top_k`, `jacobian_check_csv`) as documented in `docs/user-guide/dynamics-simulation.md`.

## Commit & Pull Request Guidelines
- No strict commit convention is enforced in history; use concise, imperative summaries (e.g., “Fix bus type handling”).
- PRs should describe motivation, key changes, and testing performed (e.g., `make test-fast`).
- Link related issues when applicable; include screenshots or plots if changes affect outputs or docs.

## Optional: Documentation
- Docs are built with MkDocs (`make docs` / `make docs-serve`).
- Keep API docs and tutorials aligned with any public behavior changes.
