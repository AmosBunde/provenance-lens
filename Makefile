.DEFAULT_GOAL := help
.PHONY: help setup lint test data features eda baseline-smoke reason eval demo

help: ## List the available targets
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the package with dev tooling and the pre-commit hooks
	python3 -m pip install -e ".[dev]"
	@if [ -f .pre-commit-config.yaml ]; then pre-commit install; fi

data: ## Download and verify sources, then build the deduplicated manifest
	python3 -m provenance_lens.data.download configs/data_sources.yaml data/raw
	python3 -m provenance_lens.data.manifest
	python3 -m provenance_lens.data.splits

features: ## Run all forensic extractors into the parquet feature store
	python3 -m provenance_lens.forensics.store

eda: ## Execute the EDA notebook headlessly into docs/report/eda/
	mkdir -p docs/report/eda
	papermill notebooks/eda.ipynb docs/report/eda/eda_executed.ipynb --kernel python3

baseline-smoke: ## CPU smoke run of the full baseline path (tiny subset)
	python3 -m provenance_lens.baseline.train --smoke

reason: ## Run the reasoner over the validation split (test only via harness)
	python3 -m provenance_lens.reasoner.batch val

eval: ## Score all tracks, calibrate, and generate docs/report/
	python3 -m provenance_lens.eval.report

PORT ?= 8000
HOST ?= 127.0.0.1
demo: ## Serve the verdict endpoint and page (HOST=127.0.0.1 PORT=8000)
	uvicorn provenance_lens.demo.app:app --host $(HOST) --port $(PORT)

lint: ## Run ruff and black in check mode
	ruff check src tests
	black --check src tests

test: ## Run the unit test suite
	pytest
