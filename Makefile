.PHONY: install test run
install:
	python -m pip install -e '.[dev]'

test:
	pytest

run:
	divar-scanner run --config config/fatemi.yaml
