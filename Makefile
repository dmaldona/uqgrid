# Simple Makefile for UQGrid development

.PHONY: help install install-dev install-petsc test test-fast clean lint coverage ci docs docs-serve docs-deploy

help:
	@echo "Available commands:"
	@echo "  install       - Install package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  install-petsc - Install with PETSc support"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run fast tests only (skip adjoint tests)"
	@echo "  lint          - Run static analysis checks"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  ci            - Run local continuous-integration suite"
	@echo "  docs          - Build MkDocs site (requires [docs] extras)"
	@echo "  docs-serve    - Serve docs locally with live reload"
	@echo "  docs-deploy   - Deploy docs to GitHub Pages"
	@echo "  clean         - Clean build artifacts"

# Installation
install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

install-petsc:
	python3 -m pip install -e ".[petsc]"

# Testing
test:
	python3 -m pytest

test-fast:
	python3 -m pytest -m "not adjoint"

lint:
	python3 -m ruff check .
	python3 -m mypy uqgrid/simulation/config.py uqgrid/simulation/pflow.py

coverage:
	python3 -m pytest --cov=uqgrid --cov-report=term-missing

ci: lint coverage

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy --force

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete