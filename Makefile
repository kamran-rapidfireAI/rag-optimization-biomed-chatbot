.PHONY: install install-dev test test-unit test-integration lint format typecheck serve clean help

# Default target
.DEFAULT_GOAL := help

# Python and package management
PYTHON := python3
PIP := $(PYTHON) -m pip

# Source directories
SRC_DIR := src
TEST_DIR := tests

help: ## Show this help message
	@echo "BioRAG Bench - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install || true

install-all: ## Install package with all optional dependencies
	$(PIP) install -e ".[all]"
	pre-commit install || true

test: ## Run all tests with coverage
	pytest $(TEST_DIR) --cov=$(SRC_DIR)/biorag --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only
	pytest $(TEST_DIR)/integration -v -m integration

test-fast: ## Run tests excluding slow ones
	pytest $(TEST_DIR) -v -m "not slow"

lint: ## Run ruff linting
	ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run ruff linting with auto-fix
	ruff check $(SRC_DIR) $(TEST_DIR) --fix

format: ## Run ruff formatting
	ruff format $(SRC_DIR) $(TEST_DIR)

format-check: ## Check formatting without making changes
	ruff format $(SRC_DIR) $(TEST_DIR) --check

typecheck: ## Run mypy type checking
	mypy $(SRC_DIR)/biorag

check: lint format-check typecheck ## Run all checks (lint, format, typecheck)

serve: ## Start the FastAPI server
	uvicorn biorag.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

demo: ## Start the Gradio demo
	$(PYTHON) demo/app.py

build-corpus: ## Build the corpus from HuggingFace datasets
	biorag build-corpus

index: ## Build FAISS index from corpus
	biorag index-faiss

eval: ## Run evaluation on golden suite
	biorag eval

sweep: ## Run hyperparameter sweep
	biorag sweep

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-data: ## Clean generated data (use with caution!)
	rm -rf data/processed/*
	rm -rf runs/*

# Development shortcuts
dev: install-dev ## Alias for install-dev

all: install-dev lint format typecheck test ## Run full development pipeline

