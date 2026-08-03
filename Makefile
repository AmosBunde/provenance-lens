.DEFAULT_GOAL := help
.PHONY: help setup lint test data

help: ## List the available targets
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the package with dev tooling and the pre-commit hooks
	python3 -m pip install -e ".[dev]"
	@if [ -f .pre-commit-config.yaml ]; then pre-commit install; fi

data: ## Download configured sources, verify checksums, extract to data/raw
	python3 -m provenance_lens.data.download configs/data_sources.yaml data/raw

lint: ## Run ruff and black in check mode
	ruff check src tests
	black --check src tests

test: ## Run the unit test suite
	pytest
