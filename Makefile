.PHONY: install test lint format typecheck build clean release-dry

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy echotrace/ --ignore-missing-imports

build:
	python -m build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

release-dry:
	make clean
	make test
	make build
	twine check dist/*
	echo "✓ Release dry run passed. Ready to tag."
