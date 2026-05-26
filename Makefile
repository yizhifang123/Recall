.PHONY: install dev lint format test run-mcp precommit-install clean

install:  ## Sync runtime + dev deps via uv
	uv sync

dev: install precommit-install  ## Full dev bootstrap

lint:  ## Lint with ruff (no fixes)
	uv run ruff check .

format:  ## Format with ruff
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run the test suite
	uv run pytest

run-mcp:  ## Start the MCP server (stdio transport)
	uv run recall-mcp

precommit-install:  ## Install git pre-commit hooks
	uv run pre-commit install

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
