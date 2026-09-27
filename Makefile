# Developer entrypoints. Windows users: run these under Git Bash, or invoke the
# underlying commands directly (see README).
#
# `make` itself is not installed on the machine this was developed on, so these
# targets are thin wrappers that have not been executed as targets. Every command
# they wrap has been run directly, and CI invokes the commands rather than the
# Makefile, so nothing depends on this file being correct.
#
# Python 3.10 is pinned deliberately — SHAP/XGBoost wheels on 3.13+ are
# unreliable. See ADR-0004.

PY := py -3.10
VENV := .venv
BIN := $(VENV)/Scripts
PYTHON := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup install lint format typecheck test test-fast test-ml test-integration \
        check clean data eda notebooks fairness drift format-check \
        compose-up compose-down migrate-roundtrip

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

data: ## Download OULAD, build the population, generate the synthetic cohort
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/build_dataset.py
	$(PYTHON) scripts/generate_synthetic_cohort.py

eda: ## Regenerate every EDA table and figure in reports/
	$(PYTHON) scripts/run_eda.py

features: ## Build the feature matrix and the train/validation/test split
	$(PYTHON) scripts/build_features.py

leakage: ## Run the as-of property test and allowlist guards on their own
	$(PYTHON) -m pytest tests/ml -m ml -q

baselines: ## Train the Phase 5 baselines and write reports/baseline_comparison.md
	$(PYTHON) scripts/train_baselines.py

model: ## Tune, calibrate and register the primary XGBoost model
	$(PYTHON) scripts/train_model.py

explain: ## Global importance, attribution stability, and worked examples
	$(PYTHON) scripts/run_explainability.py

fairness: ## Per-subgroup error rates with intervals -> reports/fairness_audit.md
	$(PYTHON) scripts/run_fairness_audit.py

drift: ## Train vs test cohort drift -> reports/drift_report.md
	$(PYTHON) scripts/run_drift_report.py --pooled

api: ## Run the API locally with reload
	$(BIN)/uvicorn backend.app.main:app --reload --port 8000

openapi: ## Regenerate docs/openapi.json from the live app
	$(PYTHON) -c "import json,pathlib; from backend.app.main import app; 		pathlib.Path('docs/openapi.json').write_text(json.dumps(app.openapi(), indent=2), encoding='utf-8')"
	@echo "wrote docs/openapi.json"

test-api: ## API tests only
	$(PYTHON) -m pytest tests/api -q

migrate: ## Apply database migrations
	$(BIN)/alembic -c backend/alembic.ini upgrade head

db: ## Migrate, seed, and score the cohort
	$(MAKE) migrate
	$(PYTHON) scripts/seed_db.py
	$(PYTHON) scripts/score_cohort.py

test-db: ## Database tests (SQLite by default; set TEST_DATABASE_URL for PostgreSQL)
	$(PYTHON) -m pytest tests/integration -m integration -q

frontend-install: ## Install frontend dependencies
	cd frontend && npm ci

frontend: ## Run the dashboard dev server (proxies /api to localhost:8000)
	cd frontend && npm run dev

frontend-check: ## Lint, typecheck, test and build the frontend
	cd frontend && npm run lint && npm run typecheck && npm test && npm run build

notebooks: ## Execute the EDA notebooks to check they still run
	# Not part of CI: these need data/raw/oulad/, which is not committed.
	# Run after `make data`.
	$(PYTHON) -m nbconvert --to notebook --execute --stdout \
		--ExecutePreprocessor.timeout=900 notebooks/*.ipynb > /dev/null
	@echo "notebooks execute cleanly"

compose-up: ## Build and run the full stack (needs Docker; never run by the author)
	docker compose up --build

compose-down: ## Stop the stack and remove volumes
	docker compose down -v

migrate-roundtrip: ## Prove migrations reverse: upgrade -> downgrade -> upgrade
	$(BIN)/alembic -c backend/alembic.ini upgrade head
	$(BIN)/alembic -c backend/alembic.ini downgrade base
	$(BIN)/alembic -c backend/alembic.ini upgrade head

check: lint format-check typecheck test ## Everything CI runs

format-check: ## Fail if formatting is off (CI runs this; `make format` fixes it)
	$(BIN)/ruff format --check .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
