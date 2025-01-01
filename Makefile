SHELL := /bin/bash

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

run: ## Run the Python main entry point.
	uv run src/main.py
	
ruff: ## Run ruff as a linter.
	uv run ruff check src/
