# Simple Makefile for documentation generation and testing

.PHONY: docs test

docs:
	uv run pdoc --math --docformat google -t docs/templates/ src/bootstrap -o docs/site/

test:
	uv run pytest
