# Simple Makefile for documentation generation and testing

.PHONY: docs test

docs:
	uv run pdoc --math --docformat google src/bootstrap -o docs/site/

test:
	uv run pytest
