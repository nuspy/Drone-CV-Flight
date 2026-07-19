.PHONY: install lint test unit integration e2e ci

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

unit:
	pytest tests/unit

integration:
	pytest tests/integration -m "not e2e"

e2e:
	pytest tests/e2e -m e2e

test: unit integration

ci: lint unit integration e2e
