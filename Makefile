# =============================================================================
# Personal AI Agent Assistant — Makefile
# =============================================================================

.PHONY: help setup install install-dev lint format typecheck test test-cov clean

PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
BLACK   := $(VENV)/bin/black
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: install-dev  ## Create venv and install all dependencies (run first!)
	@echo "✅ Environment ready. Activate with: source $(VENV)/bin/activate"
	@echo "📋 Copy your env file: cp .env.example .env"

install:  ## Install production dependencies only
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install  ## Install production + dev dependencies
	$(PIP) install -r requirements-dev.txt
	$(VENV)/bin/pre-commit install

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint:  ## Run ruff linter
	$(RUFF) check core/ skills/ main.py tests/

format:  ## Auto-format code with black + ruff
	$(BLACK) core/ skills/ main.py tests/
	$(RUFF) check --fix core/ skills/ main.py tests/

typecheck:  ## Run mypy static type checker
	$(MYPY) core/ main.py

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:  ## Run all tests
	$(PYTEST) tests/ -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=core --cov=skills --cov-report=term-missing --cov-report=html
	@echo "📊 HTML coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

run:  ## Start the CLI assistant (interactive)
	$(VENV)/bin/python main.py chat

ping:  ## Ping the configured LLM provider
	$(VENV)/bin/python main.py ping

config:  ## Show active configuration
	$(VENV)/bin/python main.py config show

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove venv, cache, and build artifacts
	rm -rf $(VENV) __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
