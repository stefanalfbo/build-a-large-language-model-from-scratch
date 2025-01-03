SHELL := /bin/bash

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

sync: ## Sync the dependencies.
	uv sync

run: ## Run the Python main entry point.
	uv run src/main.py
	
pytest: ## Run the Python tests.
	uv run pytest

linter: ## Run ruff as a linter.
	uv run ruff check .

format: ## Run ruff format to format the code.
	uv run ruff format .

isort: ## Run ruff to sort the imports.
	uv run ruff check --select I --fix .   