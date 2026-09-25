# Developer entrypoints. Windows users: run these under Git Bash, or invoke the
# underlying commands directly (see README).
#
# Python 3.10 is pinned deliberately — SHAP/XGBoost wheels on 3.13+ are
# unreliable. See ADR-0004.

PY := py -3.10
VENV := .venv
BIN := $(VENV)/Scripts
PYTHON := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup install lint format typecheck test test-fast test-ml test-integration check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install everything
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(MAKE) install

install: ## Install the package in editable mode with all extras
	$(PYTHON) -m pip install -e ".[dev,viz,api]"

lint: ## Run ruff
	$(BIN)/ruff check .

format: ## Auto-format and fix what ruff can
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Run mypy
	$(BIN)/mypy

test: ## Run the full suite with coverage
	$(PYTHON) -m pytest --cov --cov-report=term-missing

test-fast: ## Unit tests only, no slow/integration markers
	$(PYTHON) -m pytest tests/unit -m "not slow and not integration"

test-ml: ## Model and leakage tests
	$(PYTHON) -m pytest tests/ml -m ml

test-integration: ## Requires a live PostgreSQL instance
	$(PYTHON) -m pytest tests/integration -m integration

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
