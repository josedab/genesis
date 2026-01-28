.PHONY: help install dev-install lint format test test-fast coverage docs docs-serve clean build

# Default target
help:
	@echo "Genesis Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install package in editable mode"
	@echo "  dev-install   Install with all dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run all linters (ruff, mypy)"
	@echo "  format        Format code (black, isort, ruff fix)"
	@echo "  check         Run format check + lint (CI mode)"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run tests excluding slow ones"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-int      Run integration tests only"
	@echo "  coverage      Run tests with coverage report"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  docs-serve    Serve documentation locally"
	@echo ""
	@echo "Build:"
	@echo "  build         Build package"
	@echo "  clean         Remove build artifacts"

# Setup
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,pytorch]"
	pre-commit install

# Code Quality
lint:
	ruff check genesis tests
	mypy genesis --ignore-missing-imports

format:
	black genesis tests examples
	isort genesis tests
	ruff check --fix genesis tests

check:
	black --check genesis tests
	isort --check genesis tests
	ruff check genesis tests
	mypy genesis --ignore-missing-imports

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-parallel:
	pytest tests/ -v -n auto -m "not slow"

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

coverage:
	pytest tests/ --cov=genesis --cov-report=html --cov-report=term-missing -m "not slow"
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Build
build: clean
	python -m build
	twine check dist/*

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
