# Simple Makefile for UQGrid development

.PHONY: help install install-dev install-petsc test test-fast clean

help:
	@echo "Available commands:"
	@echo "  install       - Install package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  install-petsc - Install with PETSc support"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run fast tests only (skip adjoint tests)"
	@echo "  clean         - Clean build artifacts"

# Installation
install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

install-petsc:
	python -m pip install -e ".[petsc]"

# Testing
test:
	python -m pytest

test-fast:
	python -m pytest -m "not adjoint"

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete