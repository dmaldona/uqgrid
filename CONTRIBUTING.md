# Contributing to UQGrid

Thank you for your interest in contributing to UQGrid! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting up the development environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dmaldona/uqgrid.git
   cd uqgrid
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode:**
   ```bash
   make dev-setup
   ```
   
   Or manually:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

### Optional Dependencies

- **PETSc support:** For adjoint sensitivity analysis
  ```bash
  pip install -e ".[petsc]"
  ```

- **All dependencies:**
  ```bash
  pip install -e ".[all]"
  ```

## Development Workflow

### Code Quality

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated checks

Run all quality checks:
```bash
make lint
make format
```

### Testing

We have different types of tests:

- **Fast tests** (default): `make test-fast`
- **All tests**: `make test`
- **Adjoint tests** (slow): `make test-adjoint`

The adjoint tests require PETSc and are marked with `@pytest.mark.adjoint`. They are skipped by default due to their computational cost.

### Running Tests

```bash
# Run fast tests only
make test-fast

# Run all tests including slow ones
make test

# Run only adjoint validation tests
pytest --adjoint-tests
```